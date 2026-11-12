"""The prompt the model sees, and helpers to turn items into chat examples.

Kept short on purpose: every token here is paid on every call.
"""
from __future__ import annotations

import json

SYSTEM = """You cut a messy description of a process into parts. Reply with one JSON object only:
{"status":"ok|partial|refuse","reason":null|"too_long"|"sequential"|"underspecified"|"out_of_domain"|"conflicting",
"elements":[{"id":"e1","text":"short name of an activity or thing in the text"}],
"parts":[{"id":"p1","name":"...","summary":"one sentence","owns":["e1"]}],
"links":[{"from":"p1","to":"p2","uses":"e3"|null,"kind":"uses|order"}],
"missing":[{"name":"...","why":"..."}],"unsure":[{"part":"p1","why":"..."}],"unresolved":["e9"]}
Rules: every element is owned by exactly one part. A part uses another part's element only through a link, and "uses" must be an element the "to" part owns. Use kind "order" with uses null when one part must happen before another. List parts the process clearly needs but never mentions under missing. If ownership of an element is genuinely disputed, put it in unresolved and set status partial, reason conflicting. Refuse a plain step-by-step sequence (sequential), text too vague to cut (underspecified), or text that is not a process (out_of_domain); a refusal has empty lists."""

MAX_INPUT_CHARS = 6000  # about 1.5k tokens; longer inputs are refused in code


def dumps(cut: dict) -> str:
    return json.dumps(cut, ensure_ascii=False, separators=(",", ":"))


def messages(text: str, target: dict | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    if target is not None:
        msgs.append({"role": "assistant", "content": dumps(target)})
    return msgs


TOO_LONG = {"status": "refuse", "reason": "too_long", "elements": [], "parts": [], "links": [],
            "missing": [], "unsure": [], "unresolved": []}
