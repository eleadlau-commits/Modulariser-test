"""Turn a model's raw reply into a checked cut.

The model is wrapped, not trusted: inputs that are too long are refused in
code, replies are parsed and hard-checked, a failing reply is retried once
with sampling, and if that fails too the cut is returned as a refusal with
the hard-fail list attached so it is never shown as a good cut.

`generate(messages, temperature) -> str` is supplied by the backend
(vLLM on Kaggle, llama.cpp in the demo, anything else later).
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .check import hard_checks
from .prompt import MAX_INPUT_CHARS, TOO_LONG, messages

Generate = Callable[[list[dict], float], str]


def parse(raw: str):
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def cut_for(text: str, generate: Generate, retries: int = 1) -> dict:
    """Best checked cut for a text, plus diagnostics under `_meta`."""
    if len(text) > MAX_INPUT_CHARS:
        return {**TOO_LONG, "_meta": {"attempts": 0, "fails": []}}
    msgs = messages(text)
    fails_all = []
    for attempt in range(retries + 1):
        raw = generate(msgs, 0.0 if attempt == 0 else 0.7)
        cut = parse(raw)
        fails = ["unparseable JSON"] if cut is None else hard_checks(cut)
        if not fails:
            cut["_meta"] = {"attempts": attempt + 1, "fails": fails_all}
            return cut
        fails_all.append(fails[:5])
    return {"status": "refuse", "reason": "underspecified", "elements": [], "parts": [], "links": [],
            "missing": [], "unsure": [], "unresolved": [],
            "_meta": {"attempts": retries + 1, "fails": fails_all, "wrapper_refused": True}}


def strip_meta(cut: dict) -> dict:
    return {k: v for k, v in cut.items() if not k.startswith("_")}
