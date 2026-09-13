"""Direct OpenAI HTTP helper: propose structured intent only; never author answers."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .schemas import Detection


ALLOWED_INTENT_KINDS = {
    "UNOBSERVABLE",
    "OFF_TOPIC",
    "UNSUPPORTED",
    "COUNT_CATEGORY",
    "COUNT_SUBTYPE",
    "COUNT_MULTI",
    "PRESENCE_CATEGORY",
    "PRESENCE_SUBTYPE",
    "MOST_COMMON",
    "HIGHEST_CONFIDENCE",
    "LIST",
    "NEGATION",
    "EXCLUSION",
}


def _timeout_seconds() -> float:
    raw = os.environ.get("OPENAI_TIMEOUT_S", "12")
    try:
        return float(raw)
    except ValueError:
        return 12.0


def propose_structured_intent(
    question: str,
    detections: list[Detection],
    allowed_categories: list[str],
) -> tuple[dict[str, Any] | None, str]:
    """Ask the provider for JSON intent. Returns (proposal_or_None, backend_label)."""
    if os.environ.get("RUNWAYGUARD_LLM", "1").strip() in {"0", "false", "False"}:
        return None, "deterministic"
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "deterministic"

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    timeout_s = _timeout_seconds()
    evidence = [
        {
            "evidence_id": index,
            "class_name": item.class_name,
            "confidence": item.confidence,
        }
        for index, item in enumerate(detections)
    ]
    schema_hint = {
        "intent_kind": sorted(ALLOWED_INTENT_KINDS),
        "categories": allowed_categories,
        "subtype": "string|null",
        "evidence_ids": "list[int] subset of evidence_id",
        "exclude_categories": "list[str]",
    }
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
                    f"Schema hint: {json.dumps(schema_hint)}"
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
) -> dict[str, Any] | None:
    if not proposal:
        return None
    kind = proposal.get("intent_kind")
    if kind not in ALLOWED_INTENT_KINDS:
        return None
    evidence_ids = proposal.get("evidence_ids", list(range(len(detections))))
    if not isinstance(evidence_ids, list):
        return None
    if any(not isinstance(i, int) or i < 0 or i >= len(detections) for i in evidence_ids):
        return None
    categories = proposal.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]
    if not isinstance(categories, list):
        return None
    if any(cat not in allowed_categories for cat in categories):
        return None
    exclude = proposal.get("exclude_categories") or []
    if isinstance(exclude, str):
        exclude = [exclude]
    if not isinstance(exclude, list) or any(cat not in allowed_categories for cat in exclude):
        return None
    subtype = proposal.get("subtype")
    if subtype is not None and not isinstance(subtype, str):
        return None
    return {
        "intent_kind": kind,
        "categories": categories,
        "exclude_categories": exclude,
        "subtype": subtype,
        "evidence_ids": evidence_ids,
    }
