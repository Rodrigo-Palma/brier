import pytest
from fastapi.testclient import TestClient

from brier.api import app, provide_decider, provide_settings
from brier.decide import Decider
from brier.encoder import EncoderError
from brier.features import feature_size
from brier.model import initialise, with_temperature
from tests.conftest import FAKE_DIMENSION, FakeEncoder

RELEVANCE = {
    "name": "relevance",
    "kind": "bool",
    "prompt": "Does the passage answer the question?",
    "options": ["no", "yes"],
}


class BrokenEncoder(FakeEncoder):
    def encode(self, texts):
        raise EncoderError("ollama did not answer")


def _client(decider: Decider) -> TestClient:
    app.dependency_overrides[provide_decider] = lambda: decider
    return TestClient(app)


@pytest.fixture
def client():
    parameters = with_temperature(
        initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0), 1.3
    )
    yield _client(Decider(encoder=FakeEncoder(), parameters=parameters, min_confidence=0.0))
    app.dependency_overrides.clear()


def test_health_is_reachable_without_weights():
    with TestClient(app) as bare:
        assert bare.get("/health").json() == {"status": "ok"}


def test_one_request_answers_several_questions(client):
    response = client.post(
        "/decide",
        json={
            "state": "revenue grew to 94.9 billion dollars",
            "questions": [
                RELEVANCE,
                {
                    "name": "tone",
                    "kind": "score",
                    "prompt": "How does the passage read?",
                    "options": ["negative", "neutral", "positive"],
                },
            ],
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert [answer["name"] for answer in body["answers"]] == ["relevance", "tone"]
    assert body["temperature"] == pytest.approx(1.3)


def test_the_answer_carries_the_confidence_and_the_full_distribution(client):
    response = client.post("/decide", json={"state": "revenue grew", "questions": [RELEVANCE]})

    answer = response.json()["answers"][0]
    assert answer["value"] in ("no", "yes")
    assert len(answer["probabilities"]) == 2
    assert 0.0 <= answer["confidence"] <= 1.0


def test_a_low_confidence_answer_comes_back_marked_as_an_abstention():
    parameters = with_temperature(
        initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0), 1.0
    )
    client = _client(Decider(encoder=FakeEncoder(), parameters=parameters, min_confidence=0.99))

    response = client.post("/decide", json={"state": "revenue grew", "questions": [RELEVANCE]})

    assert response.json()["answers"][0]["abstained"] is True
    app.dependency_overrides.clear()


def test_a_question_with_one_option_is_rejected_before_the_model_runs(client):
    response = client.post(
        "/decide",
        json={"state": "revenue grew", "questions": [{**RELEVANCE, "options": ["yes"]}]},
    )

    assert response.status_code == 422


def test_an_empty_state_is_rejected(client):
    response = client.post("/decide", json={"state": "", "questions": [RELEVANCE]})

    assert response.status_code == 422


def test_repeated_options_are_reported_as_a_bad_request(client):
    response = client.post(
        "/decide",
        json={"state": "revenue grew", "questions": [{**RELEVANCE, "options": ["yes", "yes"]}]},
    )

    assert response.status_code == 400
    assert "repeat" in response.json()["detail"]


def test_an_unreachable_encoder_is_a_service_error_not_a_crash():
    parameters = initialise(feature_size(FAKE_DIMENSION), hidden_size=8, seed=0)
    client = _client(Decider(encoder=BrokenEncoder(), parameters=parameters))

    response = client.post("/decide", json={"state": "revenue grew", "questions": [RELEVANCE]})

    assert response.status_code == 503
    assert "ollama did not answer" in response.json()["detail"]
    app.dependency_overrides.clear()


def test_repeated_question_names_are_rejected_before_the_model_runs(client):
    """Two answers of the same name: a caller keying by name loses one."""
    response = client.post(
        "/decide",
        json={
            "state": "revenue grew",
            "questions": [RELEVANCE, {**RELEVANCE, "prompt": "Something else?"}],
        },
    )

    assert response.status_code == 422
    assert "unique" in response.text


def test_ready_reports_what_the_instance_is_serving(client):
    response = client.get("/ready")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert "temperature" in body and "weights" in body


def test_ready_says_503_when_there_are_no_weights(tmp_path, monkeypatch):
    """The sibling service already did this; without it the answer was a traceback."""
    provide_decider.cache_clear()
    provide_settings.cache_clear()
    monkeypatch.setenv("BRIER_WEIGHTS_PATH", str(tmp_path / "absent.npz"))
    app.dependency_overrides.clear()

    with TestClient(app) as bare:
        response = bare.get("/ready")

    assert response.status_code == 503
    assert "run scripts/train_model.py" in response.json()["detail"]
    provide_decider.cache_clear()
    provide_settings.cache_clear()
