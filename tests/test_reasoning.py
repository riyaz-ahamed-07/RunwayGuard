from app.reasoning import answer_question, route_question
from app.schemas import BoundingBox, Detection


def detection(name: str, confidence: float) -> Detection:
    return Detection(
        class_id=0,
        class_name=name,
        confidence=confidence,
        box=BoundingBox(x1=10, y1=20, x2=30, y2=40),
    )


def test_unobservable_safety_question_abstains() -> None:
    response = answer_question("Is the runway safe to reopen?")
    assert response.route == "UNOBSERVABLE"
    assert response.status == "INSUFFICIENT_INFORMATION"


def test_runway_clear_question_abstains() -> None:
    assert route_question("Is the runway clear?") == "UNOBSERVABLE"


def test_unrelated_question_skips_detector() -> None:
    assert route_question("What is the capital of France?") == "NO_DETECTION_NEEDED"


def test_count_confident_objects() -> None:
    response = answer_question(
        "How many debris objects are visible?",
        [detection("fastener_hardware", 0.91), detection("flexible_debris", 0.82)],
    )
    assert response.status == "ANSWERED"
    assert "2" in response.answer


def test_low_confidence_candidate_forces_abstention() -> None:
    response = answer_question(
        "How many debris objects are visible?",
        [detection("fastener_hardware", 0.91), detection("flexible_debris", 0.31)],
    )
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.guardrail_reason is not None


def test_missing_requested_class_does_not_claim_absence() -> None:
    response = answer_question("Is there a screw present?", [detection("hand_tool", 0.91)])
    assert response.route == "DETECT"
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert "not proof" in response.answer.lower()


def test_longest_alias_prefers_screwdriver_over_tool() -> None:
    response = answer_question(
        "Is there a screwdriver present?",
        [detection("hand_tool", 0.93)],
    )
    assert response.status == "ANSWERED"
    assert "screwdriver" in response.answer.lower() or "hand tool" in response.answer.lower()


def test_bounding_box_corners_are_ordered() -> None:
    box = BoundingBox(x1=40, y1=50, x2=10, y2=20)
    assert box.x1 == 10 and box.x2 == 40
    assert box.y1 == 20 and box.y2 == 50
