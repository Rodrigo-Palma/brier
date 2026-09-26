import json

import httpx
import numpy as np
import pytest

import brier.encoder as encoder_module
from brier.encoder import CachedEncoder, EncoderError, OllamaEncoder
from tests.conftest import FakeEncoder


def _posting(handler):
    """Replace httpx.post with one that routes through a MockTransport."""

    def post(url, json, timeout):  # noqa: ARG001 - mirrors the real signature
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.post(url, json=json)

    return post


def test_the_cache_asks_the_inner_encoder_once_per_text(tmp_path, encoder):
    cached = CachedEncoder(encoder, tmp_path / "cache")

    cached.encode(("alpha", "beta"))
    cached.encode(("alpha", "beta"))

    assert encoder.calls == [("alpha", "beta")]


def test_the_cache_only_asks_for_what_is_missing(tmp_path, encoder):
    cached = CachedEncoder(encoder, tmp_path / "cache")
    cached.encode(("alpha",))

    cached.encode(("alpha", "beta"))

    assert encoder.calls == [("alpha",), ("beta",)]


def test_cached_vectors_match_the_inner_encoder(tmp_path, encoder):
    direct = encoder.encode(("alpha", "beta"))
    cached = CachedEncoder(FakeEncoder(), tmp_path / "cache")

    np.testing.assert_allclose(cached.encode(("alpha", "beta")), direct)


def test_a_repeated_text_comes_back_in_the_order_it_was_asked(tmp_path, encoder):
    cached = CachedEncoder(encoder, tmp_path / "cache")

    vectors = cached.encode(("alpha", "beta", "alpha"))

    assert vectors.shape[0] == 3
    np.testing.assert_allclose(vectors[0], vectors[2])


def test_the_cache_survives_a_restart(tmp_path):
    first = FakeEncoder()
    CachedEncoder(first, tmp_path / "cache").encode(("alpha",))

    second = FakeEncoder()
    CachedEncoder(second, tmp_path / "cache").encode(("alpha",))

    assert second.calls == []


def test_encoding_nothing_is_a_mistake_not_an_empty_array(tmp_path, encoder):
    with pytest.raises(ValueError, match="nothing to encode"):
        CachedEncoder(encoder, tmp_path / "cache").encode(())


def test_an_unreachable_ollama_names_the_url():
    unreachable = OllamaEncoder(base_url="http://localhost:1")

    with pytest.raises(EncoderError, match="did not answer"):
        unreachable.encode(("alpha",))


def test_vectors_come_back_normalised(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})

    monkeypatch.setattr(encoder_module.httpx, "post", _posting(handler))

    vectors = OllamaEncoder(base_url="http://ollama.test").encode(("alpha", "beta"))

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)


def test_the_dimension_is_read_from_the_model_once(monkeypatch):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.0] * 7]})

    monkeypatch.setattr(encoder_module.httpx, "post", _posting(handler))
    encoder = OllamaEncoder(base_url="http://ollama.test")

    assert encoder.dimension == 7
    assert encoder.dimension == 7
    assert len(seen) == 1


def test_a_short_reply_is_an_error_rather_than_a_silent_truncation(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    monkeypatch.setattr(encoder_module.httpx, "post", _posting(handler))

    with pytest.raises(EncoderError, match="asked for 2 vectors, got 1"):
        OllamaEncoder(base_url="http://ollama.test").encode(("alpha", "beta"))


def test_an_http_error_is_wrapped(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(encoder_module.httpx, "post", _posting(handler))

    with pytest.raises(EncoderError, match="did not answer"):
        OllamaEncoder(base_url="http://ollama.test").encode(("alpha",))
