from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .runtime_config import ASK_ANSWER_CONFIDENCE, CLASS_NAMES
from .schemas import AskResponse, Detection, ImageSize

CONFIDENT_THRESHOLD = ASK_ANSWER_CONFIDENCE


class IntentKind(str, Enum):
    UNOBSERVABLE = "UNOBSERVABLE"
    OFF_TOPIC = "OFF_TOPIC"
    UNSUPPORTED = "UNSUPPORTED"
    COUNT_CATEGORY = "COUNT_CATEGORY"
    COUNT_SUBTYPE = "COUNT_SUBTYPE"
    COUNT_MULTI = "COUNT_MULTI"
    PRESENCE_CATEGORY = "PRESENCE_CATEGORY"
    PRESENCE_SUBTYPE = "PRESENCE_SUBTYPE"
    MOST_COMMON = "MOST_COMMON"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    LIST = "LIST"
    NEGATION = "NEGATION"
    EXCLUSION = "EXCLUSION"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    categories: tuple[str, ...] = ()
    subtype: str | None = None
    exclude_categories: tuple[str, ...] = ()


@dataclass
class AnswerFacts:
    status: str
    answer: str
    evidence: list[Detection] = field(default_factory=list)
    guardrail_reason: str | None = None
    intent: str = ""


# Fine labels / materials the 7-class head cannot certify.
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
    "plastic": "plastic_paper_debris",
    "paper": "plastic_paper_debris",
    "label": "plastic_paper_debris",
    "soda can": "component_container",
    "fuel cap": "component_container",
    "battery": "component_container",
    "pen": "component_container",
    "rock": "natural_debris",
    "wood": "natural_debris",
}

IRREGULAR_SINGULAR = {
    "batteries": "battery",
    "people": "person",
    "persons": "person",
    "knives": "knife",
}

CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "fastener_hardware": ("fastener hardware", "fastener", "hardware", "fasteners"),
    "hand_tool": ("hand tool", "hand tools", "tool", "tools"),
    "flexible_debris": ("flexible debris",),
    "loose_metal": ("loose metal",),
    "plastic_paper_debris": ("plastic paper debris",),
    "component_container": ("component container", "component", "container"),
    "natural_debris": ("natural debris",),
}

UNKNOWN_TARGETS = {
    "people",
    "person",
    "persons",
    "human",
    "humans",
    "animal",
    "animals",
    "vehicle",
    "car",
    "plane",
    "aircraft",
    "colour",
    "color",
    "colours",
    "colors",
    "weight",
    "temperature",
    "brand",
    "logo",
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

UNSUPPORTED_SPATIAL = (
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


def _singularize_token(token: str) -> str:
    if token in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 1 and not token.endswith("ss"):
        return token[:-1]
    return token


def _phrase_forms(phrase: str) -> set[str]:
    phrase = phrase.lower().strip()
    forms = {phrase}
    tokens = phrase.split()
    if len(tokens) == 1:
        forms.add(_singularize_token(phrase))
        if not phrase.endswith("s"):
            forms.add(phrase + "s")
        if phrase.endswith("y") and len(phrase) > 1:
            forms.add(phrase[:-1] + "ies")
    else:
        head, tail = tokens[:-1], tokens[-1]
        for form in _phrase_forms(tail):
            forms.add(" ".join([*head, form]))
    return {f for f in forms if f}


def _contains_phrase(text: str, phrase: str) -> bool:
    for form in _phrase_forms(phrase):
        pattern = r"(?<!\w)" + re.escape(form) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


def _humanized(name: str) -> str:
    return name.replace("_", " ")


def _match_all_subtypes(normalized: str) -> list[tuple[str, str]]:
    ranked = sorted(SUBTYPE_TO_CATEGORY.items(), key=lambda item: -len(item[0]))
    hits: list[tuple[str, str]] = []
    for subtype, category in ranked:
        if _contains_phrase(normalized, subtype):
            hits.append((subtype, category))
    return hits


def _match_all_categories(normalized: str) -> list[str]:
    ranked: list[tuple[int, str, str]] = []
    for class_name in CLASS_NAMES:
        phrases = {_humanized(class_name), *CATEGORY_ALIASES.get(class_name, ())}
        for phrase in phrases:
            ranked.append((len(phrase), class_name, phrase))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    found: list[str] = []
    for _, class_name, phrase in ranked:
        if class_name in found:
            continue
        if _contains_phrase(normalized, phrase):
            found.append(class_name)
    return found


def _mentions_unknown_target(normalized: str) -> bool:
    return any(_contains_phrase(normalized, term) for term in UNKNOWN_TARGETS)


def parse_intent(question: str) -> Intent:
    normalized = normalize(question)
    if any(term in normalized or _contains_phrase(normalized, term) for term in UNOBSERVABLE_TERMS):
        return Intent(IntentKind.UNOBSERVABLE)
    if any(_contains_phrase(normalized, term) or term in normalized for term in UNSUPPORTED_SPATIAL):
        return Intent(IntentKind.UNSUPPORTED)
    if _contains_phrase(normalized, "colour") or _contains_phrase(normalized, "color"):
        return Intent(IntentKind.UNSUPPORTED)
    if re.search(r"\bexcept\b|\bexcluding\b|\bbut not\b", normalized):
        cats = _match_all_categories(normalized)
        if cats:
            return Intent(IntentKind.EXCLUSION, exclude_categories=tuple(cats))
        return Intent(IntentKind.UNSUPPORTED)
    if re.search(r"\bno\b.+\b|\bnot any\b|\baren'?t there\b|\bisn'?t there\b", normalized) and (
        "tool" in normalized or "debris" in normalized or _match_all_categories(normalized)
    ):
        return Intent(IntentKind.NEGATION, categories=tuple(_match_all_categories(normalized)))

    subtypes = _match_all_subtypes(normalized)
    categories = _match_all_categories(normalized)
    wants_count = "how many" in normalized or _contains_phrase(normalized, "count")
    wants_presence = any(
        _contains_phrase(normalized, term) for term in ("present", "visible", "is there", "are there", "is this", "is that")
    )
    wants_most_common = "most common" in normalized
    wants_highest = "highest confidence" in normalized or "highest-confidence" in normalized
    wants_list = any(term in normalized for term in ("list", "what debris", "what objects"))

    if wants_highest:
        return Intent(IntentKind.HIGHEST_CONFIDENCE)
    if wants_most_common:
        return Intent(IntentKind.MOST_COMMON)
    if wants_count and _mentions_unknown_target(normalized) and not subtypes and not categories:
        return Intent(IntentKind.UNSUPPORTED)
    if wants_count and subtypes and not categories:
        subtype, parent = subtypes[0]
        return Intent(IntentKind.COUNT_SUBTYPE, categories=(parent,), subtype=subtype)
    if wants_count and subtypes and categories:
        # Prefer subtype when both match (battery vs component).
        subtype, parent = subtypes[0]
        return Intent(IntentKind.COUNT_SUBTYPE, categories=(parent,), subtype=subtype)
    if wants_count and len(categories) > 1:
        return Intent(IntentKind.COUNT_MULTI, categories=tuple(categories))
    if wants_count and len(categories) == 1:
        return Intent(IntentKind.COUNT_CATEGORY, categories=(categories[0],))
    if wants_count and _mentions_unknown_target(normalized):
        return Intent(IntentKind.UNSUPPORTED)
    if wants_count and any(term in normalized for term in ("debris", "object", "objects", "fod")):
        return Intent(IntentKind.COUNT_CATEGORY)
    if wants_count:
        return Intent(IntentKind.UNSUPPORTED)
    if wants_presence and subtypes:
        subtype, parent = subtypes[0]
        return Intent(IntentKind.PRESENCE_SUBTYPE, categories=(parent,), subtype=subtype)
    if wants_presence and categories:
        return Intent(IntentKind.PRESENCE_CATEGORY, categories=(categories[0],))
    if wants_list:
        return Intent(IntentKind.LIST)
    if categories or subtypes or any(term in normalized for term in ("debris", "fod", "runway", "taxiway")):
        return Intent(IntentKind.LIST)
    if _mentions_unknown_target(normalized):
        return Intent(IntentKind.UNSUPPORTED)
    return Intent(IntentKind.OFF_TOPIC)


def intent_from_validated_proposal(proposal: dict) -> Intent | None:
    try:
        kind = IntentKind(proposal["intent_kind"])
    except (KeyError, ValueError):
        return None
    return Intent(
        kind=kind,
        categories=tuple(proposal.get("categories") or ()),
        subtype=proposal.get("subtype"),
        exclude_categories=tuple(proposal.get("exclude_categories") or ()),
    )


def route_question(question: str) -> str:
    intent = parse_intent(question)
    if intent.kind == IntentKind.UNOBSERVABLE:
        return "UNOBSERVABLE"
    if intent.kind == IntentKind.OFF_TOPIC:
        return "NO_DETECTION_NEEDED"
    return "DETECT"


def _subtype_explanation(subtype: str, parent: str) -> str:
    return (
        f"'{subtype}' is finer-grained than the model's '{_humanized(parent)}' class, "
        "so identity/count for that subtype cannot be confirmed."
    )


def render_facts(intent: Intent, detections: list[Detection]) -> AnswerFacts:
    confident = [item for item in detections if item.confidence >= ASK_ANSWER_CONFIDENCE]
    uncertain = [item for item in detections if item.confidence < ASK_ANSWER_CONFIDENCE]
    counts = Counter(item.class_name for item in confident)

    if intent.kind == IntentKind.UNSUPPORTED:
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            "Insufficient information. That question targets something outside the supported FOD intents "
            "(unknown object type, unsupported property, exclusion phrasing, or spatial relation).",
            detections,
            "Unsupported intent.",
            intent.kind.value,
        )
    if intent.kind == IntentKind.NEGATION:
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            "Insufficient information. Negated presence questions are not answered as yes/no by this API; "
            "a non-detection also cannot prove absence.",
            detections,
            "Negation is not auto-answered.",
            intent.kind.value,
        )
    if intent.kind == IntentKind.EXCLUSION:
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            "Insufficient information. Exclusion phrasing ('except' / 'excluding') is not supported; "
            "ask for an explicit category count instead.",
            detections,
            "Exclusion intent abstained.",
            intent.kind.value,
        )
    if intent.kind in {IntentKind.COUNT_SUBTYPE, IntentKind.PRESENCE_SUBTYPE}:
        parent = intent.categories[0] if intent.categories else "debris"
        subtype = intent.subtype or "that subtype"
        parent_count = counts.get(parent, 0)
        extra = (
            f" I confidently detected {parent_count} {_humanized(parent)} object(s), but "
            + _subtype_explanation(subtype, parent)
            if parent_count
            else " " + _subtype_explanation(subtype, parent)
        )
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            f"Insufficient information about '{subtype}'.{extra}",
            detections,
            "Subtype outside 7-class ontology.",
            intent.kind.value,
        )
    if not confident and intent.kind not in {IntentKind.UNSUPPORTED, IntentKind.NEGATION, IntentKind.EXCLUSION}:
        reason = "Only low-confidence candidates were produced." if uncertain else "No confident detections."
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            "Insufficient information. The model has no confident visible-debris detection in this image. "
            "This must not be interpreted as proof that the runway is clear.",
            uncertain,
            reason,
            intent.kind.value,
        )
    if intent.kind == IntentKind.COUNT_MULTI:
        parts = [f"{counts[cat]} {_humanized(cat)}" for cat in intent.categories]
        answer = "Confident counts by requested class: " + "; ".join(parts) + "."
        answer += " This is not proof of absence for any requested class."
        if uncertain:
            return AnswerFacts(
                "INSUFFICIENT_INFORMATION",
                answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
                detections,
                "Low-confidence candidates could change the answer.",
                intent.kind.value,
            )
        return AnswerFacts("ANSWERED", answer, confident, None, intent.kind.value)
    if intent.kind == IntentKind.COUNT_CATEGORY:
        if not intent.categories:
            n = len(confident)
            answer = (
                f"I confidently detected {n} visible debris object(s). "
                "This is not proof that the scene is clear of all debris."
            )
            if uncertain:
                return AnswerFacts(
                    "INSUFFICIENT_INFORMATION",
                    answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
                    detections,
                    "Low-confidence candidates could change the answer.",
                    intent.kind.value,
                )
            return AnswerFacts("ANSWERED", answer, confident, None, intent.kind.value)
        cat = intent.categories[0]
        n = counts[cat]
        evidence = [item for item in confident if item.class_name == cat]
        answer = (
            f"I confidently detected {n} {_humanized(cat)} object(s). "
            "A zero or low count is not proof that none is present."
        )
        if uncertain:
            return AnswerFacts(
                "INSUFFICIENT_INFORMATION",
                answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
                detections,
                "Low-confidence candidates could change the answer.",
                intent.kind.value,
            )
        return AnswerFacts("ANSWERED", answer, evidence, None, intent.kind.value)
    if intent.kind == IntentKind.PRESENCE_CATEGORY:
        cat = intent.categories[0]
        n = counts[cat]
        if n == 0:
            return AnswerFacts(
                "INSUFFICIENT_INFORMATION",
                f"I did not confidently detect {_humanized(cat)} in this image. "
                "That is not proof that none is present.",
                detections,
                "Absence cannot be certified.",
                intent.kind.value,
            )
        return AnswerFacts(
            "ANSWERED",
            f"Yes. I confidently detected {_humanized(cat)} in the image.",
            [item for item in confident if item.class_name == cat],
            None,
            intent.kind.value,
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
            return AnswerFacts(
                "INSUFFICIENT_INFORMATION",
                answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
                detections,
                "Low-confidence candidates could change the answer.",
                intent.kind.value,
            )
        return AnswerFacts("ANSWERED", answer, confident, None, intent.kind.value)
    if intent.kind == IntentKind.HIGHEST_CONFIDENCE:
        best = max(confident, key=lambda item: item.confidence)
        answer = f"The highest-confidence detection is {_humanized(best.class_name)} at {best.confidence:.3f}."
        return AnswerFacts("ANSWERED", answer, [best], None, intent.kind.value)

    summary = ", ".join(f"{count} {_humanized(name)}" for name, count in counts.most_common())
    answer = f"Confident visible-debris detections: {summary}. Human inspection and removal are required."
    if uncertain:
        return AnswerFacts(
            "INSUFFICIENT_INFORMATION",
            answer + f" There are also {len(uncertain)} low-confidence candidate(s).",
            detections,
            "Low-confidence candidates could change the answer.",
            IntentKind.LIST.value,
        )
    return AnswerFacts("ANSWERED", answer, confident, None, IntentKind.LIST.value)


def answer_question(
    question: str,
    detections: list[Detection] | None = None,
    image_size: ImageSize | None = None,
    phrase_with_llm: bool = True,
) -> AskResponse:
    from .llm_phrase import propose_structured_intent, validate_proposal

    del image_size
    detections = detections or []
    backend = "deterministic"
    intent = parse_intent(question)

    if phrase_with_llm and intent.kind not in {IntentKind.UNOBSERVABLE, IntentKind.OFF_TOPIC}:
        proposal, backend_label = propose_structured_intent(question, detections, list(CLASS_NAMES))
        validated = validate_proposal(
            proposal,
            detections=detections,
            allowed_categories=set(CLASS_NAMES),
        )
        if validated:
            llm_intent = intent_from_validated_proposal(validated)
            if llm_intent is not None:
                intent = llm_intent
                backend = backend_label
            else:
                backend = "deterministic_fallback"
        elif backend_label != "deterministic":
            backend = "deterministic_fallback"

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

    facts = render_facts(intent, detections)
    return AskResponse(
        route="DETECT",
        status=facts.status,  # type: ignore[arg-type]
        answer=facts.answer,
        evidence=facts.evidence,
        guardrail_reason=facts.guardrail_reason,
        intent=facts.intent,
        phrasing_backend=backend,
    )

