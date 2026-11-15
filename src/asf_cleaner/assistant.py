"""Natural-language assistant for the map page.

The page understands simple commands on its own (no network). Anything else is sent here, where a Claude model
answers using only the numbers the page already shows (the `context` it sends), so it cannot invent results.
Without ANTHROPIC_API_KEY the endpoint says it is not configured and the page falls back to commands.
"""
from __future__ import annotations

import json
import os

import requests

API = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ASF_ASSISTANT_MODEL", "claude-sonnet-5-5")
MAX_QUESTION = 600
MAX_CONTEXT = 6000
LANGS = {"en": "English", "pt": "Brazilian Portuguese", "es": "Spanish"}

SYSTEM = (
    "You are the assistant inside SARIS, an observatory of surface change built on NASA NISAR radar, rain (GPM IMERG), "
    "and optical data. Answer in {language}, in at most four short sentences, for a non-specialist. Use ONLY the "
    "context JSON provided; if it does not contain the answer, say what the user can do on the map to get it "
    "(pick a point or draw an area, then run the forecast). Never invent numbers. Forecasts are statistical "
    "estimates with uncertainty, not certainties."
)


def configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def clean(question: str, context: dict | None) -> tuple[str, str]:
    """Bound the input so a page (or anyone) cannot send an enormous prompt."""
    q = (question or "").strip()[:MAX_QUESTION]
    ctx = json.dumps(context or {}, ensure_ascii=False, default=str)[:MAX_CONTEXT]
    return q, ctx


def ask(question: str, lang: str, context: dict | None, *, post=requests.post) -> str:
    q, ctx = clean(question, context)
    if not q:
        raise ValueError("empty question")
    body = {
        "model": MODEL,
        "max_tokens": 400,
        "system": SYSTEM.format(language=LANGS.get(lang, "English")),
        "messages": [{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}"}],
    }
    res = post(API, json=body, timeout=30, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    res.raise_for_status()
    return "".join(b.get("text", "") for b in res.json().get("content", []) if b.get("type") == "text").strip()
