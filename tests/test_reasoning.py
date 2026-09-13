from app.reasoning import answer_question, parse_intent, route_question
from app.schemas import BoundingBox, Detection
from app.reasoning import IntentKind


def detection(name: str, confidence: float) -> Detection:
    return Detection(
        class_id=0,
        class_name=name,
        confidence=confidence,
        box=BoundingBox(x1=10, y1=20, x2=30, y2=40),
    )


def test_unobservable_safety_question_abstains() -> None:
    response = answer_question("Is the runway safe to reopen?", phrase_with_llm=False)
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
        phrase_with_llm=False,
    )
    assert response.status == "ANSWERED"
    assert "2" in response.answer


def test_category_count_uses_only_that_class() -> None:
    dets = [detection("hand_tool", 0.91), detection("fastener_hardware", 0.88)]
    tools = answer_question("How many hand tools are visible?", dets, phrase_with_llm=False)
    assert tools.status == "ANSWERED"
    assert "1" in tools.answer
    assert "hand tool" in tools.answer


def test_bolt_subtype_count_abstains() -> None:
    dets = [detection("hand_tool", 0.91), detection("fastener_hardware", 0.88)]
    response = answer_question("How many bolts are visible?", dets, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert "bolt" in response.answer.lower()
    assert parse_intent("How many bolts are visible?").kind == IntentKind.COUNT_SUBTYPE


def test_screwdriver_presence_does_not_overclaim() -> None:
    response = answer_question(
        "Is there a screwdriver present?",
        [detection("hand_tool", 0.93)],
        phrase_with_llm=False,
    )
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert "hand tool" in response.answer.lower()
    assert "screwdriver" in response.answer.lower()


def test_low_confidence_candidate_forces_abstention() -> None:
    response = answer_question(
        "How many debris objects are visible?",
        [detection("fastener_hardware", 0.91), detection("flexible_debris", 0.31)],
        phrase_with_llm=False,
    )
    assert response.status == "INSUFFICIENT_INFORMATION"


def test_most_common_reports_ties() -> None:
    response = answer_question(
        "What is the most common object?",
        [detection("hand_tool", 0.9), detection("fastener_hardware", 0.91)],
        phrase_with_llm=False,
    )
    assert response.status == "ANSWERED"
    assert "tie" in response.answer.lower()


def test_highest_confidence_intent() -> None:
    response = answer_question(
        "Which object has the highest confidence?",
        [detection("hand_tool", 0.7), detection("fastener_hardware", 0.95)],
        phrase_with_llm=False,
    )
    assert response.status == "ANSWERED"
    assert "fastener" in response.answer.lower()


def test_spatial_question_unsupported() -> None:
    response = answer_question(
        "Is the wire on the left?",
        [detection("flexible_debris", 0.9)],
        phrase_with_llm=False,
    )
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "UNSUPPORTED"


def test_generic_present_object_question_lists_detections() -> None:
    assert route_question("what is the object that is present?") == "DETECT"
    assert parse_intent("what is the object that is present?").kind == IntentKind.LIST
    response = answer_question(
        "what is the object that is present?",
        [detection("component_container", 0.92)],
        phrase_with_llm=False,
    )
    assert response.route == "DETECT"
    assert response.status == "ANSWERED"
    assert "component container" in response.answer.lower()


def test_bounding_box_corners_are_ordered() -> None:
    box = BoundingBox(x1=40, y1=50, x2=10, y2=20)
    assert box.x1 == 10 and box.x2 == 40
