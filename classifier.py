"""Category classifier: Jev (System One via OpenRouter) -> LLM (OpenRouter) -> keyword heuristic.

Order of preference:
  1. Jev model  - via OpenRouter Decisions API (needs only OPENROUTER_API_KEY)
  2. LLM        - via OpenRouter chat completions (same key)
  3. Keyword    - offline, always works (final fallback)
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

from engine import CATEGORIES

logger = logging.getLogger("classifier")

JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

CATEGORY_DESCRIPTIONS = {
    "Groceries & Food": "supermarket, grocery store, market, food staples, cooking ingredients, bread, rice, vegetables",
    "Dining Out": "restaurant, cafe, coffee shop, bubble tea, takeaway, fast food, eating out, delivery meal",
    "Transportation": "fuel, petrol, gasoline, taxi, grab, bus, train, metro, parking, vehicle maintenance, toll",
    "Utilities & Rent": "rent, apartment, electricity, water bill, internet, mobile phone bill, gas, insurance, tv subscription",
    "Shopping & Tech": "electronics, gadgets, headphones, phone accessories, clothes, cosmetics, skincare, online shopping, home goods",
}

# keyword fallback (lowercased)
KEYWORDS: dict[str, list[str]] = {
    "Groceries & Food": ["grocer", "supermarket", "mart", "cho ", "market", "food", "bakery", "vegetable", "rice", "milk", "egg", "bread"],
    "Dining Out": ["pho", "restaurant", "cafe", "coffee", "starbucks", "tea", "sushi", "burger", "pizza", "takeaway", "delivery", "foodpanda", "grab food", "eat"],
    "Transportation": ["fuel", "gas", "petrol", "xang", "grab", "taxi", "bus", "train", "metro", "parking", "toll", "transport", "uber"],
    "Utilities & Rent": ["rent", "thue", "electricity", "dien", "water", "nuoc", "internet", "phone", "mobile", "bill", "gas", "insurance", "utility"],
    "Shopping & Tech": ["shopee", "lazada", "tiki", "uniqlo", "zara", "h&m", "sony", "apple", "iphone", "headphone", "cosmetic", "skincare", "clothes", "shoe", "bag", "watch", "gadget", "amazon"],
}


def _keyword_classify(item: str) -> str | None:
    text = item.lower()
    for cat in CATEGORIES:
        if any(kw in text for kw in KEYWORDS[cat]):
            return cat
    return None


def classify_with_jev(items: list[str]) -> dict[str, str] | None:
    """Batch-classify items with Jev via the OpenRouter Decisions API.

    One Choice question per item, all answered in parallel in a single call.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        return None
    try:
        payload = {
            "model": JEV_MODEL,
            "state": {"items": items},
            "questions": {
                f"cat_{i}": {
                    "type": "choice",
                    "instructions": f"Which single category best fits purchase [{i}] '{it}'?",
                    "criteria": {c: CATEGORY_DESCRIPTIONS[c] for c in CATEGORIES},
                }
                for i, it in enumerate(items)
            },
        }
        req = urllib.request.Request(
            DECISIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        answers = data.get("answers", {})
        result: dict[str, str] = {}
        for i, it in enumerate(items):
            ans = answers.get(f"cat_{i}") or {}
            choice = ans.get("choice") if isinstance(ans, dict) else None
            if choice in CATEGORIES:
                result[it] = choice
        logger.info("Jev classified %d/%d items via OpenRouter", len(result), len(items))
        return result or None
    except Exception as exc:  # pragma: no cover - network / auth / quota issues
        logger.warning("Jev classification failed (%s); falling back", exc)
        return None


def classify_with_llm(items: list[str]) -> dict[str, str] | None:
    """Classify items with an OpenRouter LLM; expects a JSON mapping item -> category."""
    if not os.getenv("OPENROUTER_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        cats = ", ".join(CATEGORIES)
        prompt = (
            "Classify each purchase below into exactly one of these categories:\n"
            f"{cats}\n\n"
            "Items:\n" + "\n".join(f"- {it}" for it in items) +
            "\n\nReturn ONLY a JSON object mapping each exact item string to a category, "
            "e.g. {\"item\": \"Groceries & Food\"}. No extra text."
        )
        resp = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "system", "content": "You are a precise transaction categorizer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
        import json

        mapping = json.loads(text)
        result = {str(k): str(v) for k, v in mapping.items() if str(v) in CATEGORIES}
        logger.info("LLM classified %d/%d items", len(result), len(items))
        return result or None
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM classification failed (%s); using keyword fallback", exc)
        return None


def classify_items(items: list[str], prefer_jev: bool = True) -> tuple[dict[str, str], str]:
    """Return (item -> category, method_used). Falls back Jev -> LLM -> keyword."""
    seen: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.append(it)

    result: dict[str, str] = {}
    method = "keyword"

    if prefer_jev:
        j = classify_with_jev(seen)
        if j:
            result.update(j)
            method = "jev"
    if not all(result.get(it) for it in seen):
        missing = [it for it in seen if not result.get(it)]
        llm = classify_with_llm(missing)
        if llm:
            result.update(llm)
            method = "llm" if method != "jev" else "jev+llm"
    for it in seen:
        if not result.get(it):
            kw = _keyword_classify(it)
            if kw:
                result[it] = kw

    return result, method