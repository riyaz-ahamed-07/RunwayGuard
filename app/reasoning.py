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
    "fastener_hardware": ("fastener", "hardware", "bolt", "nut", "nail", "screw", "washer", "clamp"),
    "hand_tool": ("tool", "wrench", "pliers", "cutter", "hammer", "screwdriver"),
    "flexible_debris": ("flexible debris", "wire", "cable", "hose", "tape"),
    "loose_metal": ("loose metal", "metal sheet", "metal part"),
    "plastic_paper_debris": ("plastic", "paper", "label", "luggage tag", "paint chip"),
    "component_container": ("component", "container", "battery", "fuel cap", "pen", "soda can"),
    "natural_debris": ("natural debris", "rock", "wood"),
}

UNOBSERVABLE_TERMS = {
    "safe to reopen",
    "safe for landing",
    "safe for takeoff",
    "will it damage",
    "will the plane",
    "who dropped",
    "when was",
    "how heavy",
    "exact weight",
    "what material",
    "exact distance",
}

DETECTION_TERMS = {
    "detect",
    "debris",
    "fod",
    "object",
    "objects",
    "count",
    "how many",
    "most common",
    "present",
    "visible",
    "runway",
    "taxiway",
    "bolt",
    "nut",
    "wire",
    "tool",
    "rock",
    "highest confidence",
} | {name.replace("_", " ") for name in SUPPORTED_CLASSES} | {
    alias for aliases in REQUEST_ALIASES.values() for alias in aliases
}


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().strip())


def route_question(question: str) -> str:
    normalized = normalize(question)
    if any(term in normalized for term in UNOBSERVABLE_TERMS):
        return "UNOBSERVABLE"
    if any(term in normalized for term in DETECTION_TERMS):
        return "DETECT"
    return "NO_DETECTION_NEEDED"


def _humanized(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").lower()


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
    requested_class = next(
        (
            class_name
            for class_name in SUPPORTED_CLASSES
            if _humanized(class_name) in normalized
            or class_name.lower() in normalized
            or any(alias in normalized for alias in REQUEST_ALIASES[class_name])
        ),
        None,
    )

    if "most common" in normalized:
        class_name, count = counts.most_common(1)[0]
        answer = f"The most common confidently detected debris class is {_humanized(class_name)} ({count})."
    elif "how many" in normalized or "count" in normalized:
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
    elif requested_class and any(term in normalized for term in ("present", "visible", "is there", "are there")):
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
            answer=answer + f" There are also {len(uncertain)} low-confidence candidate(s), so the complete count is uncertain.",
            evidence=detections,
            guardrail_reason="Low-confidence candidates could change the answer.",
        )

    return AskResponse(route="DETECT", status="ANSWERED", answer=answer, evidence=confident)
