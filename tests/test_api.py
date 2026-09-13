from fastapi.testclient import TestClient

from app.main import app


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
