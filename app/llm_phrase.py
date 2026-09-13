from __future__ import annotations

import os
from typing import Any

import httpx

from .schemas import Detection


def maybe_phrase_answer(
    question: str,
    allowed_conclusion: str,
    evidence: list[Detection],
    intent: str,
) -> tuple[str, str]:
    """Optional OpenAI phrasing over an already-validated conclusion.

    The model may only reword `allowed_conclusion`. It must not add objects,
    change counts, or weaken abstentions. On any failure, return the original text.
    """
    if os.environ.get("RUNWAYGUARD_LLM", "1").strip() in {"0", "false", "False"}:
        return allowed_conclusion, "deterministic"

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return allowed_conclusion, "deterministic"

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "12"))
    payload_evidence = [
        {
            "class_name": item.class_name,
            "confidence": item.confidence,
            "box": item.box.model_dump(),
        }
        for item in evidence
    ]
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rephrase an already-validated airport FOD detection answer. "
                    "Use only the allowed conclusion and evidence JSON. "
                    "Do not invent objects, classes, counts, locations, or safety claims. "
                    "Do not contradict the allowed conclusion. "
                    "Return one short plain-language paragraph."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Intent: {intent}\n"
                    f"Allowed conclusion: {allowed_conclusion}\n"
                    f"Evidence JSON: {payload_evidence}\n"
                    "Rewrite the allowed conclusion only."
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
        if not text:
            return allowed_conclusion, "deterministic"
        return text, f"openai:{model}"
    except Exception:
        return allowed_conclusion, "deterministic"
