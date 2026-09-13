from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

from .runtime_config import ASK_ANSWER_CONFIDENCE, CLASS_NAMES
from .schemas import AskResponse, Detection, ImageSize

# Back-compat alias used by API docs / health.
CONFIDENT_THRESHOLD = ASK_ANSWER_CONFIDENCE


class IntentKind(str, Enum):
    UNOBSERVABLE = "UNOBSERVABLE"
    OFF_TOPIC = "OFF_TOPIC"
    COUNT_CATEGORY = "COUNT_CATEGORY"
    COUNT_SUBTYPE = "COUNT_SUBTYPE"
    PRESENCE_CATEGORY = "PRESENCE_CATEGORY"
    PRESENCE_SUBTYPE = "PRESENCE_SUBTYPE"
    MOST_COMMON = "MOST_COMMON"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    LIST = "LIST"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    category: str | None = None
    subtype: str | None = None


SUBTYPE_TO_CATEGORY: dict[str, str] = {
    "screwdriver": "hand_tool",
    "adjustable wrench": "hand_tool",
    "wrench": "hand_tool",
    "pliers": "hand_tool",
    "cutter": "hand_tool",
    "hammer": "hand_tool",
    "bolt washer": "fastener_hardware",
    "bolt nut": "fastener_hardware",
    "bolt": "fastener_hardware",
    "nut": "fastener_hardware",
    "nail": "fastener_hardware",
    "screw": "fastener_hardware",
    "washer": "fastener_hardware",
    "clamp": "fastener_hardware",
    "hose": "flexible_debris",
    "tape": "flexible_debris",
    "wire": "flexible_debris",
    "cable": "flexible_debris",
    "metal sheet": "loose_metal",
    "metal part": "loose_metal",
    "luggage tag": "plastic_paper_debris",
    "paint chip": "plastic_paper_debris",
    "label": "plastic_paper_debris",
    "soda can": "component_container",
    "fuel cap": "component_container",
    "battery": "component_container",
    "pen": "component_container",
    "rock": "natural_debris",
    "wood": "natural_debris",
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "fastener_hardware": ("fastener hardware", "fastener", "hardware"),
    "hand_tool": ("hand tool", "hand tools", "tool", "tools"),
    "flexible_debris": ("flexible debris",),
    "loose_metal": ("loose metal",),
    "plastic_paper_debris": ("plastic paper debris", "plastic", "paper"),
    "component_container": ("component container", "component", "container"),
    "natural_debris": ("natural debris",),
}

UNOBSERVABLE_TERMS = (
    "safe to reopen",
    "safe for landing",
    "safe for takeoff",
    "runway safe",
    "taxiway safe",
    "is the runway clear",
    "is runway clear",
    "runway is clear",
    "clear to land",
    "clear for takeoff",
    "will it damage",
    "will the plane",
    "who dropped",
    "when was",
    "how heavy",
    "exact weight",
    "what material",
    "exact distance",
    "how far",
)

UNSUPPORTED_TERMS = (
    "on the left",
    "on the right",
    "to the left",
    "to the right",
    "at the top",
    "at the bottom",
    "how far apart",
    "distance between",
    "overlapping",
    "behind",
    "in front of",
)


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def _phrase_forms(phrase: str) -> set[str]:
    phrase = phrase.lower().strip()
    forms = {phrase}
    if " " in phrase:
        head, tail = phrase.rsplit(" ", 1)
        if tail.endswith("s") and len(tail) > 1:
            forms.add(f"{head} {tail[:-1]}")
        else:
            forms.add(f"{head} {tail}s")
    else:
        if phrase.endswith("s") and len(phrase) > 1:
            forms.add(phrase[:-1])
        else:
            forms.add(phrase + "s")
    return forms


def _contains_phrase(text: str, phrase: str) -> bool:
    for form in _phrase_forms(phrase):
        pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _humanized(name: str) -> str:
    return name.replace("_", " ")


def _match_subtype(normalized: str) -> tuple[str, str] | None:
    ranked = sorted(SUBTYPE_TO_CATEGORY.items(), key=lambda item: -len(item[0]))
    for subtype, category in ranked:
        if _contains_phrase(normalized, subtype):
            return subtype, category
    return None


def _match_category(normalized: str) -> str | None:
    ranked: list[tuple[int, str, str]] = []
    for class_name in CLASS_NAMES:
        phrases = {_humanized(class_name), *CATEGORY_ALIASES.get(class_name, ())}
        for phrase in phrases:
            ranked.append((len(phrase), class_name, phrase))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    for _, class_name, phrase in ranked:
        if _contains_phrase(normalized, phrase):
            return class_name
    return None


def parse_intent(question: str) -> Intent:
    normalized = normalize(question)
    if any(term in normalized or _contains_phrase(normalized, term) for term in UNOBSERVABLE_TERMS):
        return Intent(IntentKind.UNOBSERVABLE)
    if any(_contains_phrase(normalized, term) or term in normalized for term in UNSUPPORTED_TERMS):
        return Intent(IntentKind.UNSUPPORTED)

    subtype_hit = _match_subtype(normalized)
    category_hit = _match_category(normalized)
    wants_count = "how many" in normalized or _contains_phrase(normalized, "count")
    wants_presence = any(
        _contains_phrase(normalized, term) for term in ("present", "visible", "is there", "are there")
    )
    wants_most_common = "most common" in normalized
    wants_highest = "highest confidence" in normalized or "highest-confidence" in normalized
    wants_list = any(term in normalized for term in ("list", "what debris", "what objects", "detect"))

    if wants_highest:
        return Intent(IntentKind.HIGHEST_CONFIDENCE)
    if wants_most_common:
        return Intent(IntentKind.MOST_COMMON)
    if wants_count and subtype_hit:
        return Intent(IntentKind.COUNT_SUBTYPE, category=subtype_hit[1], subtype=subtype_hit[0])
    if wants_count and category_hit:
        return Intent(IntentKind.COUNT_CATEGORY, category=category_hit)
    if wants_count:
        return Intent(IntentKind.COUNT_CATEGORY)
    if wants_presence and subtype_hit:
        return Intent(IntentKind.PRESENCE_SUBTYPE, category=subtype_hit[1], subtype=subtype_hit[0])
    if wants_presence and category_hit:
        return Intent(IntentKind.PRESENCE_CATEGORY, category=category_hit)
    if wants_list or category_hit or subtype_hit or any(
        term in normalized for term in ("debris", "fod", "runway", "taxiway", "object", "objects")
    ):
        return Intent(IntentKind.LIST)
    return Intent(IntentKind.OFF_TOPIC)


def route_question(question: str) -> str:
    intent = parse_intent(question)
    if intent.kind == IntentKind.UNOBSERVABLE:
        return "UNOBSERVABLE"
    if intent.kind == IntentKind.OFF_TOPIC:
        return "NO_DETECTION_NEEDED"
    return "DETECT"


def _conclusion_from_detections(
    intent: Intent,
    detections: list[Detection],
) -> tuple[str, str, list[Detection], str | None]:
    confident = [item for item in detections if item.confidence >= ASK_ANSWER_CONFIDENCE]
    uncertain = [item for item in detections if item.confidence < ASK_ANSWER_CONFIDENCE]

    if intent.kind == IntentKind.UNSUPPORTED:
        return (
            "INSUFFICIENT_INFORMATION",
            "Insufficient information. That question needs spatial or relational reasoning this API does not support.",
            detections,
            "Unsupported intent.",
        )

    if not confident:
        reason = (
            "Only low-confidence candidates were produced."
            if uncertain
            else "No debris was detected above the answer confidence threshold."
        )
        return (
            "INSUFFICIENT_INFORMATION",
            (
                "Insufficient information. The model has no confident visible-debris detection in this image. "
                "This must not be interpreted as proof that the runway is clear."
            ),
            uncertain,
            reason,
        )

    counts = Counter(item.class_name for item in confident)

    if intent.kind == IntentKind.COUNT_SUBTYPE:
        parent = intent.category or "debris"
        parent_count = counts.get(parent, 0)
        subtype = intent.subtype or "that subtype"
        extra = (
            f" I do see {parent_count} confident {_humanized(parent)} detection(s), "
            f"but the seven-class model cannot confirm a specific '{subtype}'."
            if parent_count
            else f" The seven-class model cannot identify a specific '{subtype}'."
        )
        return (
            "INSUFFICIENT_INFORMATION",
            f"Insufficient information about '{subtype}'.{extra}",
            detections,
            "Subtype identity is outside the 7-class ontology.",
        )

    if intent.kind == IntentKind.PRESENCE_SUBTYPE:
        parent = intent.category or "debris"
        parent_count = counts.get(parent, 0)
        subtype = intent.subtype or "that subtype"
        if parent_count:
            answer = (
                f"Insufficient information. I confidently detected {_humanized(parent)}, "
                f"but that operational class is broader than '{subtype}' "
                f"(for example a hammer and a screwdriver share hand_tool)."
            )
        else:
            answer = (
                f"Insufficient information. I did not confidently detect {_humanized(parent)}, "
                f"and cannot confirm '{subtype}'. That is not proof none is present."
            )
        return (
            "INSUFFICIENT_INFORMATION",
            answer,
            detections,
            "Subtype identity is outside the 7-class ontology.",
        )

    if intent.kind == IntentKind.COUNT_CATEGORY:
        if intent.category:
            n = counts[intent.category]
            evidence = [item for item in confident if item.class_name == intent.category]
            if uncertain:
                return (
                    "INSUFFICIENT_INFORMATION",
                    (
                        f"I confidently detected {n} {_humanized(intent.category)} object(s), "
                        f"but {len(uncertain)} low-confidence candidate(s) make a complete count uncertain."
                    ),
                    detections,
                    "Low-confidence candidates could change the answer.",
                )
            return (
                "ANSWERED",
                f"I confidently detected {n} {_humanized(intent.category)} object(s).",
                evidence,
                None,
            )
        if uncertain:
            return (
                "INSUFFICIENT_INFORMATION",
                (
                    f"I confidently detected {len(confident)} visible debris object(s), "
                    f"but {len(uncertain)} low-confidence candidate(s) make a complete count uncertain."
                ),
                detections,
                "Low-confidence candidates could change the answer.",
            )
        return ("ANSWERED", f"I confidently detected {len(confident)} visible debris object(s).", confident, None)

    if intent.kind == IntentKind.PRESENCE_CATEGORY and intent.category:
        n = counts[intent.category]
        if n == 0:
            return (
                "INSUFFICIENT_INFORMATION",
                (
                    f"I did not confidently detect {_humanized(intent.category)} in this image. "
                    "That is not proof that none is present."
                ),
                detections,
                "Absence cannot be certified from a detector non-detection.",
            )
        return (
            "ANSWERED",
            f"Yes. I confidently detected {_humanized(intent.category)} in the image.",
            [item for item in confident if item.class_name == intent.category],
            None,
        )

    if intent.kind == IntentKind.MOST_COMMON:
        top = counts.most_common()
        best = top[0][1]
        tied = [name for name, count in top if count == best]
        if len(tied) > 1:
            labels = ", ".join(_humanized(name) for name in tied)
            answer = f"There is a tie for most common confident class: {labels} ({best} each)."
        else:
            answer = f"The most common confidently detected debris class is {_humanized(tied[0])} ({best})."
        if uncertain:
            return (
                "INSUFFICIENT_INFORMATION",
                answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
                detections,
                "Low-confidence candidates could change the answer.",
            )
        return ("ANSWERED", answer, confident, None)

    if intent.kind == IntentKind.HIGHEST_CONFIDENCE:
        best = max(confident, key=lambda item: item.confidence)
        answer = (
            f"The highest-confidence detection is {_humanized(best.class_name)} "
            f"at {best.confidence:.3f}."
        )
        return ("ANSWERED", answer, [best], None)

    summary = ", ".join(f"{count} {_humanized(name)}" for name, count in counts.most_common())
    answer = f"Confident visible-debris detections: {summary}. Human inspection and removal are required."
    if uncertain:
        return (
            "INSUFFICIENT_INFORMATION",
            answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
            detections,
            "Low-confidence candidates could change the answer.",
        )
    return ("ANSWERED", answer, confident, None)


def answer_question(
    question: str,
    detections: list[Detection] | None = None,
    image_size: ImageSize | None = None,
    phrase_with_llm: bool = True,
) -> AskResponse:
    from .llm_phrase import maybe_phrase_answer

    del image_size
    intent = parse_intent(question)
    if intent.kind == IntentKind.UNOBSERVABLE:
        return AskResponse(
            route="UNOBSERVABLE",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                "Insufficient information. One image can support detection of visible candidate debris only; "
                "it cannot certify runway safety, predict aircraft damage, or determine hidden physical facts. "
                "An authorized inspection is still required."
            ),
            evidence=[],
            guardrail_reason="The question asks for information that is not observable from one image.",
            intent=intent.kind.value,
            phrasing_backend="deterministic",
        )
    if intent.kind == IntentKind.OFF_TOPIC:
        return AskResponse(
            route="NO_DETECTION_NEEDED",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                "This API answers questions about visible airport foreign-object debris. "
                "The question is outside that supported scope."
            ),
            evidence=[],
            guardrail_reason="No supported image-detection intent was identified.",
            intent=intent.kind.value,
            phrasing_backend="deterministic",
        )

    detections = detections or []
    status, answer, evidence, guardrail = _conclusion_from_detections(intent, detections)
    phrased, backend = answer, "deterministic"
    if phrase_with_llm and status == "ANSWERED":
        phrased, backend = maybe_phrase_answer(question, answer, evidence, intent.kind.value)
        if not phrased.strip() or len(phrased) > 1200:
            phrased, backend = answer, "deterministic"
    return AskResponse(
        route="DETECT",
        status=status,  # type: ignore[arg-type]
        answer=phrased if status == "ANSWERED" else answer,
        evidence=evidence,
        guardrail_reason=guardrail,
        intent=intent.kind.value,
        phrasing_backend=backend,
    )
