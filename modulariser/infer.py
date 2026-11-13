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


def _num(i: str) -> int:
    try:
        return int(i[1:])
    except ValueError:
        return 10**9


def repair(cut: dict) -> dict | None:
    """Deterministic fixes for the common structural slips, or None if hopeless.

    Leaks keep their first owner, orphans join the part owning the nearest
    element before them (ids follow the text), links whose `uses` lives in
    another part are re-pointed at that part, dangling references are dropped.
    The cut's content (elements, names, grouping) is otherwise left alone.
    """
    if not isinstance(cut, dict) or not isinstance(cut.get("elements"), list) or not isinstance(cut.get("parts"), list):
        return None
    c = {k: cut.get(k) for k in ("status", "reason", "elements", "parts", "links", "missing", "unsure", "unresolved")}
    for k in ("links", "missing", "unsure", "unresolved"):
        c[k] = list(c[k] or [])
    seen, els = set(), []
    for e in c["elements"]:
        if isinstance(e, dict) and e.get("id") not in seen:
            seen.add(e.get("id"))
            els.append(e)
    c["elements"] = els
    known = [e["id"] for e in els]
    if c["status"] == "refuse":
        c["reason"] = c["reason"] or "underspecified"
        return c
    owner: dict[str, str] = {}
    parts = []
    for p in c["parts"]:
        owns = []
        for e in p.get("owns", []):
            if e in seen and e not in owner:
                owner[e] = p["id"]
                owns.append(e)
        parts.append({**p, "owns": owns})
    unresolved = [e for e in dict.fromkeys(c["unresolved"]) if e in seen and e not in owner]
    for e in sorted(set(known) - set(owner) - set(unresolved), key=_num):
        before = [o for o in owner if _num(o) < _num(e)]
        after = [o for o in owner if _num(o) > _num(e)]
        ref = max(before, key=_num) if before else (min(after, key=_num) if after else None)
        if ref is None:
            return None
        owner[e] = owner[ref]
        next(p for p in parts if p["id"] == owner[ref])["owns"].append(e)
    parts = [p for p in parts if p["owns"]]
    pids = {p["id"] for p in parts}
    links, lseen = [], set()
    for l in c["links"]:
        if not isinstance(l, dict) or l.get("from") not in pids:
            continue
        to, uses = l.get("to"), l.get("uses")
        if uses is not None:
            if uses not in owner:
                continue
            to = owner[uses]
        if to not in pids or to == l["from"]:
            continue
        key = (l["from"], to, uses, l.get("kind"))
        if key not in lseen:
            lseen.add(key)
            links.append({"from": l["from"], "to": to, "uses": uses, "kind": l.get("kind", "uses")})
    c.update(parts=parts, links=links, unresolved=unresolved,
             unsure=[u for u in c["unsure"] if isinstance(u, dict) and u.get("part") in pids])
    if unresolved and c["status"] == "ok":
        c["status"] = "partial"
    if c["status"] == "partial" and c["reason"] is None:
        c["reason"] = "conflicting"
    if c["status"] == "ok":
        c["reason"] = None
    return c


def cut_for(text: str, generate: Generate, retries: int = 1, use_repair: bool = True) -> dict:
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
            cut["_meta"] = {"attempts": attempt + 1, "fails": fails_all, "repaired": False}
            return cut
        fails_all.append(fails[:5])
        fixed = repair(cut) if (use_repair and cut is not None) else None
        if fixed is not None and not hard_checks(fixed):
            fixed["_meta"] = {"attempts": attempt + 1, "fails": fails_all, "repaired": True}
            return fixed
    return {"status": "refuse", "reason": "underspecified", "elements": [], "parts": [], "links": [],
            "missing": [], "unsure": [], "unresolved": [],
            "_meta": {"attempts": retries + 1, "fails": fails_all, "wrapper_refused": True}}


def strip_meta(cut: dict) -> dict:
    return {k: v for k, v in cut.items() if not k.startswith("_")}
