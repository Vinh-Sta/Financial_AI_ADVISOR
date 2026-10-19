"""LLM orchestration layer - strict financial coach via OpenRouter.

Prompt design follows The Five Principles of Prompting:
  1. Give Direction  - strict coach persona + explicit mission
  2. Specify Format  - fixed MARKDOWN section headings (natural text, not JSON)
  3. Provide Examples- one worked example of the expected output
  4. Evaluate Quality- self-check instruction before answering
  5. Divide Labor    - deterministic math stays in the engine (code);
                       the LLM only judges, prescribes and critiques

Free models produce noticeably worse text when forced into JSON mode, so the
coach writes natural markdown. If the model still returns JSON, we parse it
tolerantly and render it structurally - best of both worlds.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("advisor")

SYSTEM_PROMPT = """You are caring personal financial coach.

Mission: help the user reach their savings goal. You never sugarcoat: you call out
overspending bluntly, you assign concrete cut targets, and you tell the user straight
when their goal is unrealistic at the current pace.

Rules:
- Base every number you mention on the data provided. Never invent figures.
- Be specific: point out unnecessary spending and suggested measures.
- Keep the tone firm, warm at the end, zero fluff.

Write your advice as MARKDOWN TEXT with EXACTLY these four sections (use these exact headings):

## Diagnosis
1-2 sentences: what the numbers say about the user's finances right now.

## Mandatory interventions
A bullet list of 3 main concrete actions. Each bullet states the action, the category it applies to, and the expected monthly saving.

## Timeline critique
1-2 sentences: is the goal achievable in the target months? If not, say so and give the realistic projection.

## Motivation
1 sentences of honest encouragement tied to a real milestone.

Example:
## Diagnosis
You save 2.1M/month on average but Dining Out jumped 68% last month, erasing half your gains.

## Mandatory interventions
- Cut dining-out to 2 meals/week (Dining Out) - saves about 450,000/month
- Cancel the streaming subscriptions you never watch (Utilities & Rent) - saves about 150,000/month
- Meal-prep Sundays so groceries replace takeaway (Groceries & Food) - saves about 300,000/month

## Timeline critique
At the current pace you need 26 months for a 12-month goal. Not achievable without cuts.

## Motivation
You already banked 31M - the hardest part is done. Lock in the dining cut and you are back on track.

Quality check before you answer:
- Are the numbers consistent with the data? (no invented figures)
- Is every intervention specific enough to act on tomorrow?
- Did you tell the truth about the timeline, even if it hurts?
"""


# --------------------------------------------------------------------------- parsing

def get_advice(json_context: str, model: str = "openrouter/free") -> dict[str, Any]:
    """Call OpenRouter. Returns {ok, data|None, structured: bool, raw}.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return {"ok": False, "error": "Missing OPENROUTER_API_KEY in .env", "raw": ""}

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Analyze this user's financial context and give your strict coaching advice.\n\n"
                    f"CONTEXT (JSON):\n{json_context}"
                ),
            },
        ],
        temperature=0.5,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content or ""
    return {"ok": True, "data": None, "structured": False, "raw": raw}
