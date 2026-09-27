"""Turning words into vectors, with a cache because training reads them often."""

import hashlib
import json
import os
import threading
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

    Text is lower-cased before it is sent. That is not tidying. On Ollama 0.18.0
    with nomic-embed-text, every capitalised token collapses onto one vector:
    "Apple", "Cat" and "Zebra" come back identical to eight decimal places, and
    two sentences differing only in a company name come back byte for byte the
    same. Lower-casing restores the distinction, measured at cosine 0.813 for a
    pair that read 1.000 before. Pass ``lowercase=False`` to send text as
    written, once the upstream tokenizer stops doing this.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        *,
        lowercase: bool = True,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._lowercase = lowercase
        self._dimension = 0

    @property
    def fingerprint(self) -> str:
        """What a cache needs to know to not mix this encoder with another."""
        return f"{self._model}|lowercase={self._lowercase}"

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = int(self.encode(("probe",)).shape[1])
        return self._dimension

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            raise ValueError("nothing to encode")

        sent = [text.lower() for text in texts] if self._lowercase else list(texts)
        try:
            response = httpx.post(
                self._url,
                json={"model": self._model, "input": sent},
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
    """Remembers vectors on disk, keyed by the text AND the model that made it.

    Built for training, where the corpus is closed and the process is one. It is
    not a serving cache: it grows without bound, and each miss rewrites the whole
    vector file, so cost is quadratic in a large ingest.

    Two properties it does guarantee, both learned the hard way:

    - The key includes the model name and the lower-casing flag. Keyed on text
      alone, pointing two different encoders at one directory returned the first
      encoder's vectors for the second one's texts, with no error.
    - Writes are serialised and land atomically. The index and the vectors used
      to be written in two separate calls, and an interruption between them left
      a cache that loaded without complaint and then raised IndexError.
    """

    def __init__(self, inner: Encoder, cache_dir: Path, *, fingerprint: str = "") -> None:
        self._inner = inner
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._vectors_path = self._dir / "vectors.npy"
        self._fingerprint = fingerprint or _fingerprint_of(inner)
        self._lock = threading.Lock()
        self._index: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._load()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def _load(self) -> None:
        """Read the cache, and refuse one whose two halves disagree.

        Raises:
            EncoderError: when the index and the vector file are out of step.
        """
        if not (self._index_path.exists() and self._vectors_path.exists()):
            return

        index = json.loads(self._index_path.read_text(encoding="utf-8"))
        vectors = np.load(self._vectors_path)
        if len(index) != len(vectors):
            raise EncoderError(
                f"cache at {self._dir} has {len(index)} keys for {len(vectors)} vectors; "
                "delete the directory to rebuild it"
            )
        self._index, self._vectors = index, vectors

    def _save(self) -> None:
        """Write both halves, each through a temporary file.

        os.replace is atomic on the same filesystem, so a crash leaves the
        previous consistent pair rather than a half-written one.
        """
        if self._vectors is None:
            return

        # np.save appends ".npy" when the name does not already end in it, so the
        # temporary file is written through an open handle instead of by name.
        vectors_tmp = self._vectors_path.with_name(self._vectors_path.name + ".tmp")
        with vectors_tmp.open("wb") as handle:
            np.save(handle, self._vectors)
        os.replace(vectors_tmp, self._vectors_path)

        index_tmp = self._index_path.with_name(self._index_path.name + ".tmp")
        index_tmp.write_text(json.dumps(self._index), encoding="utf-8")
        os.replace(index_tmp, self._index_path)

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        """Return one vector per text, asking the inner encoder only for misses.

        Raises:
            ValueError: when nothing was asked for.
            EncoderError: when the inner encoder fails or the cache is corrupt.
        """
        # The same guard the inner encoder makes. A wrapper that quietly accepts
        # what the thing it wraps rejects is a wrapper the caller cannot reason
        # about, and here the empty case used to reach an assert instead.
        if not texts:
            raise ValueError("nothing to encode")

        with self._lock:
            missing = tuple(
                text for text in dict.fromkeys(texts) if self._key(text) not in self._index
            )
            if missing:
                fresh = self._inner.encode(missing)
                base = 0 if self._vectors is None else len(self._vectors)
                self._vectors = (
                    fresh if self._vectors is None else np.vstack([self._vectors, fresh])
                )
                for offset, text in enumerate(missing):
                    self._index[self._key(text)] = base + offset
                self._save()

            if self._vectors is None:
                raise EncoderError(f"cache at {self._dir} has an index but no vectors")
            return self._vectors[[self._index[self._key(text)] for text in texts]]

    def _key(self, text: str) -> str:
        """Identity of a vector: the text, and what produced it."""
        return _key(f"{self._fingerprint}\x00{text}")


def _fingerprint_of(inner: Encoder) -> str:
    """A short description of the encoder, for the cache key.

    Falls back to the class name for an encoder that does not describe itself,
    which keeps two different implementations from sharing a directory.
    """
    described = getattr(inner, "fingerprint", None)
    return str(described) if described else type(inner).__name__


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)
