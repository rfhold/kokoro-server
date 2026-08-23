from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class FakeAudio:
    def numpy(self) -> np.ndarray:
        return np.array([0.0, 0.5, -0.5], dtype=np.float32)


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, text: str, voice: str, speed: float):
        self.calls.append((text, voice, speed))
        return [SimpleNamespace(audio=FakeAudio())]


def test_health_reports_model_not_loaded() -> None:
    main.pipelines.clear()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "loading",
        "device": "cuda",
        "model": "hexgrad/Kokoro-82M",
        "languages": [],
    }


def test_speech_rejects_empty_input() -> None:
    response = client.post("/v1/audio/speech", json={"input": ""})

    assert response.status_code == 400
    assert response.json() == {"detail": "Input text must not be empty"}


def test_speech_reports_unavailable_pipeline() -> None:
    main.pipelines.clear()

    response = client.post("/v1/audio/speech", json={"input": "Hello"})

    assert response.status_code == 503


def test_speech_streams_pcm_from_voice_language() -> None:
    pipeline = FakePipeline()
    main.pipelines.clear()
    main.pipelines["b"] = pipeline

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello",
            "voice": "bf_emma",
            "response_format": "pcm",
            "speed": 1.25,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["content-disposition"] == "attachment; filename=speech.pcm"
    assert response.content == np.array([0, 16383, -16383], dtype=np.int16).tobytes()
    assert pipeline.calls == [("Hello", "bf_emma", 1.25)]
