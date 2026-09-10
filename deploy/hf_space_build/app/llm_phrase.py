"""Direct OpenAI HTTP helper: propose structured intent only; never author answers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .schemas import Detection

ALLOWED_KEYS = {"intent_kind", "categories", "exclude_categories", "subtype", "evidence_ids"}

ALLOWED_INTENT_KINDS = {
    "UNOBSERVABLE",
    "OFF_TOPIC",
    "UNSUPPORTED",
    "COUNT_CATEGORY",
    "COUNT_SUBTYPE",
    "COUNT_MULTI",
    "PRESENCE_CATEGORY",
    "PRESENCE_MULTI",
    "PRESENCE_SUBTYPE",
    "MOST_COMMON",
    "HIGHEST_CONFIDENCE",
    "LIST",
    "NEGATION",
    "EXCLUSION",
}

# Local parser decisions that an LLM must never weaken or replace.
PROTECTED_LOCAL_KINDS = {
    "UNOBSERVABLE",
    "OFF_TOPIC",
    "UNSUPPORTED",
    "COUNT_SUBTYPE",
    "PRESENCE_SUBTYPE",
    "NEGATION",
    "EXCLUSION",
}

CARDINALITY = {
    "UNOBSERVABLE": (0, 0),
    "OFF_TOPIC": (0, 0),
    "UNSUPPORTED": (0, 0),
    "LIST": (0, 7),
    "MOST_COMMON": (0, 0),
    "HIGHEST_CONFIDENCE": (0, 0),
    "COUNT_CATEGORY": (0, 1),  # 0 = total debris count
    "COUNT_SUBTYPE": (1, 1),
    "COUNT_MULTI": (2, 7),
    "PRESENCE_CATEGORY": (1, 1),
    "PRESENCE_MULTI": (2, 7),
    "PRESENCE_SUBTYPE": (1, 1),
    "NEGATION": (0, 7),
    "EXCLUSION": (0, 7),
}


def _timeout_seconds() -> float:
    raw = os.environ.get("OPENAI_TIMEOUT_S", "12")
    try:
        value = float(raw)
    except ValueError:
        return 12.0
    if value != value or value <= 0 or value == float("inf"):
        return 12.0
    return value


def _is_strict_int(value: object) -> bool:
    return type(value) is int and not isinstance(value, bool)


def _is_strict_str(value: object) -> bool:
    return type(value) is str


def propose_structured_intent(
    question: str,
    detections: list[Detection],
    allowed_categories: list[str],
) -> tuple[dict[str, Any] | None, str]:
    if os.environ.get("RUNWAYGUARD_LLM", "1").strip() in {"0", "false", "False"}:
        return None, "deterministic"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "deterministic"

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    timeout_s = _timeout_seconds()
    evidence = [
        {"evidence_id": index, "class_name": item.class_name, "confidence": item.confidence}
        for index, item in enumerate(detections)
    ]
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Propose ONLY a JSON object selecting intent for airport FOD Q&A. "
                    "Do not answer the user. Do not invent evidence_ids. "
                    "Use only allowed intent_kind and category names. "
                    f"Allowed kinds: {sorted(ALLOWED_INTENT_KINDS)}. "
                    "Required keys only: intent_kind, categories, exclude_categories, subtype, evidence_ids."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "evidence": evidence,
                        "allowed_categories": allowed_categories,
                    }
                ),
            },
        ],
    }
    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout_s,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        proposal = json.loads(text)
        if not isinstance(proposal, dict):
            return None, "deterministic_fallback"
        return proposal, f"openai_intent:{model}"
    except Exception:
        return None, "deterministic_fallback"


def validate_proposal(
    proposal: dict[str, Any] | None,
    *,
    detections: list[Detection],
    allowed_categories: set[str],
    subtype_to_category: dict[str, str],
) -> dict[str, Any] | None:
    """Strict per-intent schema. Returns None on any malformation."""
    try:
        if not isinstance(proposal, dict):
            return None
        if set(proposal.keys()) - ALLOWED_KEYS:
            return None
        kind = proposal.get("intent_kind")
        if not _is_strict_str(kind) or kind not in ALLOWED_INTENT_KINDS:
            return None

        categories = proposal.get("categories", [])
        if categories is None:
            categories = []
        if not isinstance(categories, list) or any(not _is_strict_str(cat) for cat in categories):
            return None
        if len(categories) != len(set(categories)):
            return None
        if any(cat not in allowed_categories for cat in categories):
            return None

        exclude = proposal.get("exclude_categories", [])
        if exclude is None:
            exclude = []
        if not isinstance(exclude, list) or any(not _is_strict_str(cat) for cat in exclude):
            return None
        if len(exclude) != len(set(exclude)):
            return None
        if any(cat not in allowed_categories for cat in exclude):
            return None

        subtype = proposal.get("subtype", None)
        if subtype is not None and not _is_strict_str(subtype):
            return None

        evidence_ids = proposal.get("evidence_ids", list(range(len(detections))))
        if evidence_ids is None:
            evidence_ids = list(range(len(detections)))
        if not isinstance(evidence_ids, list):
            return None
        if any(not _is_strict_int(i) or i < 0 or i >= len(detections) for i in evidence_ids):
            return None
        if len(evidence_ids) != len(set(evidence_ids)):
            return None

        lo, hi = CARDINALITY[kind]
        if not (lo <= len(categories) <= hi):
            return None
        if kind in {"COUNT_SUBTYPE", "PRESENCE_SUBTYPE"}:
            if subtype is None or subtype not in subtype_to_category:
                return None
            if categories[0] != subtype_to_category[subtype]:
                return None
        elif subtype is not None and kind not in {"NEGATION", "EXCLUSION", "LIST"}:
            # subtype only meaningful for subtype intents
            if kind not in {"COUNT_SUBTYPE", "PRESENCE_SUBTYPE"}:
                return None
        if kind == "EXCLUSION" and not exclude:
            return None

        return {
            "intent_kind": kind,
            "categories": categories,
            "exclude_categories": exclude,
            "subtype": subtype,
            "evidence_ids": evidence_ids,
        }
    except Exception:
        return None


def proposal_compatible_with_local(local_kind: str, local_categories: tuple[str, ...], proposal: dict[str, Any]) -> bool:
    """LLM may refine only non-protected intents and must not change operation/targets."""
    if local_kind in PROTECTED_LOCAL_KINDS:
        return False
    if proposal["intent_kind"] in PROTECTED_LOCAL_KINDS:
        return False
    if proposal["intent_kind"] != local_kind:
        return False
    if local_categories and tuple(proposal["categories"]) != local_categories:
        return False
    return True
