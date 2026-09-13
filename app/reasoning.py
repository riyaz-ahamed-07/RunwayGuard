from __future__ import annotations

import re
from collections import Counter

from .schemas import AskResponse, Detection


CONFIDENT_THRESHOLD = 0.50

SUPPORTED_CLASSES = (
    "fastener_hardware",
    "hand_tool",
    "flexible_debris",
    "loose_metal",
    "plastic_paper_debris",
    "component_container",
    "natural_debris",
)

REQUEST_ALIASES = {
    "fastener_hardware": (
        "fastener hardware",
        "fastener",
        "hardware",
        "bolt washer",
        "bolt nut",
        "washer",
        "clamp",
        "bolt",
        "nut",
        "nail",
        "screw",
    ),
    "hand_tool": (
        "hand tool",
        "screwdriver",
        "adjustable wrench",
        "wrench",
        "pliers",
        "cutter",
        "hammer",
        "tool",
    ),
    "flexible_debris": (
        "flexible debris",
        "cable",
        "hose",
        "tape",
        "wire",
    ),
    "loose_metal": (
        "loose metal",
        "metal sheet",
        "metal part",
        "metal",
    ),
    "plastic_paper_debris": (
        "plastic paper debris",
        "luggage tag",
        "paint chip",
        "plastic",
        "paper",
        "label",
    ),
    "component_container": (
        "component container",
        "soda can",
        "fuel cap",
        "battery",
        "container",
        "component",
        "pen",
    ),
    "natural_debris": (
        "natural debris",
        "rock",
        "wood",
    ),
}

UNOBSERVABLE_TERMS = {
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
    "authorize takeoff",
    "will it damage",
    "will the plane",
    "who dropped",
    "when was",
    "how heavy",
    "exact weight",
    "what material",
    "exact distance",
    "how far",
}

DETECTION_TERMS = {
    "detect",
    "debris",
    "fod",
    "count",
    "how many",
    "most common",
    "present",
    "visible",
    "highest confidence",
} | {name.replace("_", " ") for name in SUPPORTED_CLASSES} | {
    alias for aliases in REQUEST_ALIASES.values() for alias in aliases
}


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
    return re.search(pattern, text) is not None


def route_question(question: str) -> str:
    normalized = normalize(question)
    if any(_contains_phrase(normalized, term) or term in normalized for term in UNOBSERVABLE_TERMS):
        return "UNOBSERVABLE"
    if any(_contains_phrase(normalized, term) for term in DETECTION_TERMS):
        return "DETECT"
    if "runway" in normalized or "taxiway" in normalized or "object" in normalized:
        return "DETECT"
    return "NO_DETECTION_NEEDED"


def _humanized(name: str) -> str:
    return name.replace("_", " ").lower()


def _find_requested_class(normalized: str) -> str | None:
    candidates: list[tuple[int, str, str]] = []
    for class_name in SUPPORTED_CLASSES:
        phrases = {_humanized(class_name), class_name.lower(), *REQUEST_ALIASES[class_name]}
        for phrase in phrases:
            candidates.append((len(phrase), class_name, phrase))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, class_name, phrase in candidates:
        if _contains_phrase(normalized, phrase):
            return class_name
    return None


def answer_question(question: str, detections: list[Detection] | None = None) -> AskResponse:
    route = route_question(question)
    normalized = normalize(question)

    if route == "UNOBSERVABLE":
        return AskResponse(
            route="UNOBSERVABLE",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                "Insufficient information. One image can support detection of visible candidate debris only; "
                "it cannot certify runway safety, predict aircraft damage, or determine hidden physical facts. "
                "An authorized inspection is still required."
            ),
            guardrail_reason="The question asks for information that is not observable from one image.",
        )

    if route == "NO_DETECTION_NEEDED":
        return AskResponse(
            route="NO_DETECTION_NEEDED",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                "This API answers questions about visible airport foreign-object debris. "
                "The question is outside that supported scope."
            ),
            guardrail_reason="No supported image-detection intent was identified.",
        )

    detections = detections or []
    confident = [item for item in detections if item.confidence >= CONFIDENT_THRESHOLD]
    uncertain = [item for item in detections if item.confidence < CONFIDENT_THRESHOLD]

    if not confident:
        reason = (
            "Only low-confidence candidates were produced."
            if uncertain
            else "No debris was detected above the configured threshold."
        )
        return AskResponse(
            route="DETECT",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                "Insufficient information. The model has no confident visible-debris detection in this image. "
                "This must not be interpreted as proof that the runway is clear."
            ),
            evidence=uncertain,
            guardrail_reason=reason,
        )

    counts = Counter(item.class_name for item in confident)
    requested_class = _find_requested_class(normalized)

    if "most common" in normalized:
        class_name, count = counts.most_common(1)[0]
        answer = f"The most common confidently detected debris class is {_humanized(class_name)} ({count})."
    elif "how many" in normalized or _contains_phrase(normalized, "count"):
        if requested_class:
            requested_count = counts[requested_class]
            if requested_count == 0:
                return AskResponse(
                    route="DETECT",
                    status="INSUFFICIENT_INFORMATION",
                    answer=(
                        f"I did not confidently detect {_humanized(requested_class)} in this image. "
                        "That is not proof that none is present."
                    ),
                    evidence=detections,
                    guardrail_reason="Absence cannot be certified from a detector non-detection.",
                )
            answer = f"I confidently detected {requested_count} {_humanized(requested_class)} object(s)."
        else:
            answer = f"I confidently detected {len(confident)} visible debris object(s)."
    elif requested_class and any(
        _contains_phrase(normalized, term) for term in ("present", "visible", "is there", "are there")
    ):
        if counts[requested_class] == 0:
            return AskResponse(
                route="DETECT",
                status="INSUFFICIENT_INFORMATION",
                answer=(
                    f"I did not confidently detect {_humanized(requested_class)} in this image. "
                    "That is not proof that none is present."
                ),
                evidence=detections,
                guardrail_reason="Absence cannot be certified from a detector non-detection.",
            )
        answer = f"Yes. I confidently detected {_humanized(requested_class)} in the image."
    else:
        summary = ", ".join(f"{count} {_humanized(name)}" for name, count in counts.most_common())
        answer = f"Confident visible-debris detections: {summary}. Human inspection and removal are required."

    if uncertain:
        return AskResponse(
            route="DETECT",
            status="INSUFFICIENT_INFORMATION",
            answer=(
                answer
                + f" There are also {len(uncertain)} low-confidence candidate(s), so the complete count is uncertain."
            ),
            evidence=detections,
            guardrail_reason="Low-confidence candidates could change the answer.",
        )

    return AskResponse(route="DETECT", status="ANSWERED", answer=answer, evidence=confident)
