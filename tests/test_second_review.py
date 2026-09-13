from app.llm_phrase import validate_proposal
from app.reasoning import answer_question, parse_intent, render_facts, IntentKind
from app.schemas import BoundingBox, Detection


def detection(name: str, confidence: float, class_id: int = 0) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=name,
        confidence=confidence,
        box=BoundingBox(x1=10, y1=20, x2=30, y2=40),
    )


PAIR = [
    detection("hand_tool", 0.93, 1),
    detection("fastener_hardware", 0.91, 0),
]


def test_people_count_is_unsupported() -> None:
    response = answer_question("How many people are visible?", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "UNSUPPORTED"
    assert "2" not in response.answer


def test_batteries_are_subtype_not_total_count() -> None:
    response = answer_question("How many batteries are visible?", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert "battery" in response.answer.lower()
    assert parse_intent("How many batteries are visible?").kind == IntentKind.COUNT_SUBTYPE


def test_exclusion_abstains() -> None:
    response = answer_question("Count everything except hand tools", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "EXCLUSION"


def test_multi_category_count() -> None:
    response = answer_question(
        "How many hand tools and fasteners are visible?",
        PAIR,
        phrase_with_llm=False,
    )
    assert response.status == "ANSWERED"
    assert "hand tool" in response.answer.lower()
    assert "fastener" in response.answer.lower()


def test_negation_abstains() -> None:
    response = answer_question("Are there no hand tools?", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "NEGATION"
    assert not response.answer.lower().startswith("yes.")


def test_is_this_a_screwdriver_abstains() -> None:
    response = answer_question("Is this a screwdriver?", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert "screwdriver" in response.answer.lower()


def test_colour_question_unsupported() -> None:
    response = answer_question("What colour is the hand tool?", PAIR, phrase_with_llm=False)
    assert response.status == "INSUFFICIENT_INFORMATION"
    assert response.intent == "UNSUPPORTED"


def test_plastic_alias_is_subtype() -> None:
    assert parse_intent("How many plastic items are visible?").kind == IntentKind.COUNT_SUBTYPE


def test_validate_proposal_rejects_bad_evidence_and_category() -> None:
    bad = {
        "intent_kind": "COUNT_CATEGORY",
        "categories": ["spaceship"],
        "evidence_ids": [0, 99],
        "exclude_categories": [],
        "subtype": None,
    }
    assert validate_proposal(bad, detections=PAIR, allowed_categories={"hand_tool", "fastener_hardware"}) is None


def test_malicious_free_text_cannot_become_answer(monkeypatch) -> None:
    # Even if a legacy phrasing path existed, render_facts is authoritative.
    facts = render_facts(parse_intent("How many hand tools are visible?"), PAIR)
    assert facts.status == "ANSWERED"
    assert "1" in facts.answer
    assert "99" not in facts.answer
    assert "safe" not in facts.answer.lower()


def test_llm_proposal_with_invented_id_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("RUNWAYGUARD_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_propose(question, detections, allowed_categories):
        return (
            {
                "intent_kind": "COUNT_CATEGORY",
                "categories": ["hand_tool"],
                "evidence_ids": [0, 5],
                "exclude_categories": [],
                "subtype": None,
            },
            "openai_intent:fake",
        )

    monkeypatch.setattr("app.llm_phrase.propose_structured_intent", fake_propose)
    response = answer_question("How many hand tools are visible?", PAIR, phrase_with_llm=True)
    assert response.phrasing_backend == "deterministic_fallback"
    assert "1" in response.answer
