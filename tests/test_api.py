import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import reset_model_cache
from app.schemas import BoundingBox, DetectResponse, Detection, ImageSize


@pytest.fixture(autouse=True)
def _isolate_model_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    monkeypatch.setenv("RUNWAYGUARD_LLM", "0")
    reset_model_cache()
    yield
    reset_model_cache()


client = TestClient(app)


def test_ask_safety_question_without_image() -> None:
    response = client.post("/ask", data={"question": "Is the runway safe to reopen?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "UNOBSERVABLE"
    assert payload["status"] == "INSUFFICIENT_INFORMATION"
    assert "x-request-id" in response.headers


def test_ask_off_topic_without_image() -> None:
    response = client.post("/ask", data={"question": "What is the capital of France?"})
    assert response.status_code == 200
    assert response.json()["route"] == "NO_DETECTION_NEEDED"


def test_ask_detect_intent_without_image_abstains() -> None:
    response = client.post("/ask", data={"question": "How many bolts are visible?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["route"] == "DETECT"
    assert payload["status"] == "INSUFFICIENT_INFORMATION"
    assert "image" in payload["answer"].lower()


def test_health_without_weights_is_unavailable() -> None:
    response = client.get("/health")
    assert response.status_code == 503


def test_ask_ignores_caller_confidence_for_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, float] = {}

    def fake_detect(image, confidence: float):
        captured["confidence"] = confidence
        return DetectResponse(
            detections=[
                Detection(
                    class_id=0,
                    class_name="hand_tool",
                    confidence=0.91,
                    box=BoundingBox(x1=1, y1=1, x2=2, y2=2),
                ),
                Detection(
                    class_id=1,
                    class_name="fastener_hardware",
                    confidence=0.40,
                    box=BoundingBox(x1=3, y1=3, x2=4, y2=4),
                ),
            ],
            image_size=ImageSize(width=300, height=300),
            model="fake.pt",
            confidence_threshold=confidence,
            imgsz=480,
        )

    monkeypatch.setattr("app.main.detect", fake_detect)
    from PIL import Image

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(40, 40, 40)).save(image_path, format="JPEG")
    response = client.post(
        "/ask",
        data={"question": "How many debris objects are visible?", "confidence": "0.99"},
        files={"image": ("sample.jpg", image_path.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert captured["confidence"] == 0.25
    payload = response.json()
    assert payload["status"] == "INSUFFICIENT_INFORMATION"
