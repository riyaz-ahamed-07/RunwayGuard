from app.eval_ops import Box, match_detections, operating_point_table
from app.llm_phrase import proposal_compatible_with_local, validate_proposal
from app.reasoning import IntentKind, SUBTYPE_TO_CATEGORY, answer_question, parse_intent
from app.schemas import BoundingBox, Detection


def detection(name: str, confidence: float, class_id: int = 0) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=confidence,
        box=BoundingBox(x1=10, y1=20, x2=30, y2=40),
    )


TOOL = [detection("hand_tool", 0.90, 1)]
PAIR = [detection("hand_tool", 0.93, 1), detection("fastener_hardware", 0.91, 0)]
BOTH_PRESENT = [
    detection("hand_tool", 0.93, 1),
    detection("natural_debris", 0.91, 6),
]


def test_llm_cannot_override_subtype_to_category(monkeypatch) -> None:
    monkeypatch.setenv("RUNWAYGUARD_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    def fake_propose(question, detections, allowed_categories):
        return (
            {
                "intent_kind": "COUNT_CATEGORY",
                "categories": ["hand_tool"],
                "exclude_categories": [],
                "subtype": None,
                "evidence_ids": [0],
            },
            "openai_intent:fake",
        )

    monkeypatch.setattr("app.llm_phrase.propose_structured_intent", fake_propose)
    response = answer_question("How many screwdrivers?", TOOL, phrase_with_llm=True)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "COUNT_SUBTYPE"
    assert "screwdriver" in response.answer.lower()


def test_empty_presence_category_proposal_rejected() -> None:
    proposal = {
        "intent_kind": "PRESENCE_CATEGORY",
        "categories": [],
        "exclude_categories": [],
        "subtype": None,
        "evidence_ids": [0],
    }
    assert (
        validate_proposal(
            proposal,
            detections=TOOL,
            allowed_categories={"hand_tool"},
            subtype_to_category=SUBTYPE_TO_CATEGORY,
        )
        is None
    )


def test_boolean_evidence_id_rejected() -> None:
    proposal = {
        "intent_kind": "LIST",
        "categories": [],
        "exclude_categories": [],
        "subtype": None,
        "evidence_ids": [True],
    }
    assert (
        validate_proposal(
            proposal,
            detections=TOOL,
            allowed_categories={"hand_tool"},
            subtype_to_category=SUBTYPE_TO_CATEGORY,
        )
        is None
    )


def test_nested_category_rejected() -> None:
    proposal = {
        "intent_kind": "COUNT_CATEGORY",
        "categories": [{"name": "hand_tool"}],
        "exclude_categories": [],
        "subtype": None,
        "evidence_ids": [0],
    }
    assert (
        validate_proposal(
            proposal,
            detections=TOOL,
            allowed_categories={"hand_tool"},
            subtype_to_category=SUBTYPE_TO_CATEGORY,
        )
        is None
    )


def test_protected_local_blocks_compatible_check() -> None:
    proposal = {
        "intent_kind": "COUNT_CATEGORY",
        "categories": ["hand_tool"],
        "exclude_categories": [],
        "subtype": None,
        "evidence_ids": [0],
    }
    assert proposal_compatible_with_local("COUNT_SUBTYPE", ("hand_tool",), proposal) is False


def test_official_plastic_paper_debris_outranks_plastic_subtype() -> None:
    intent = parse_intent("How many plastic paper debris objects?")
    assert intent.kind == IntentKind.COUNT_CATEGORY
    assert intent.categories == ("plastic_paper_debris",)


def test_people_and_hand_tools_mixed_is_unsupported() -> None:
    response = answer_question("How many people and hand tools?", TOOL, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "UNSUPPORTED"


def test_weight_property_outranks_list_fallback() -> None:
    response = answer_question("What is the weight of the hand tool?", TOOL, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "UNSUPPORTED"


def test_multi_presence_requires_all_categories() -> None:
    missing = answer_question(
        "Are there hand tools and natural debris?",
        TOOL,
        phrase_with_llm=False,
    )
    assert missing.status == "INSUFFICIENT_INFORMATION"
    assert missing.intent == "PRESENCE_MULTI"
    assert "natural debris" in missing.answer.lower()

    both = answer_question(
        "Are there hand tools and natural debris?",
        BOTH_PRESENT,
        phrase_with_llm=False,
    )
    assert both.status == "ANSWERED"
    assert "hand tool" in both.answer.lower()
    assert "natural debris" in both.answer.lower()


def test_all_seven_category_names_parse() -> None:
    names = [
        "fastener hardware",
        "hand tool",
        "flexible debris",
        "loose metal",
        "plastic paper debris",
        "component container",
        "natural debris",
    ]
    for name in names:
        intent = parse_intent(f"How many {name} are visible?")
        assert intent.kind in {IntentKind.COUNT_CATEGORY, IntentKind.COUNT_MULTI}


def test_operating_point_synthetic_boxes() -> None:
    gt = [Box(0, 0, 10, 10, class_id=0)]
    preds = [
        Box(0, 0, 10, 10, class_id=0, confidence=0.9),
        Box(50, 50, 60, 60, class_id=1, confidence=0.4),
    ]
    low, high = operating_point_table(preds, gt, thresholds=(0.25, 0.50))
    assert low["false_positive"] == 1
    assert high["false_positive"] == 0
    assert high["true_positive"] == 1
    matched = match_detections(preds, gt, score_threshold=0.25)
    assert matched["precision"] == 0.5


def test_operating_point_class_confusion_counts_as_fp() -> None:
    gt = [Box(0, 0, 10, 10, class_id=0)]
    preds = [Box(0, 0, 10, 10, class_id=1, confidence=0.9)]
    matched = match_detections(preds, gt, score_threshold=0.25)
    assert matched["class_confusion"] == 1
    assert matched["false_positive"] == 1
    assert matched["true_positive"] == 0
    assert matched["false_negative"] == 1
    assert matched["precision"] == 0.0
    assert matched["recall"] == 0.0
