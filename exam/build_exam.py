"""Turn the compact exam sources (exam/src/*.txt) into gold items (exam/items/).

Source format, one item per block:

    === id | domain | synthetic|real | origin | licence | ok|partial|refuse
    messy input text, any number of lines
    ---
    el k1: element text
    edges: k1>k2, k2>>k3          (> uses, >> order)
    q: question | answer | k1,k2
    ch: change name | k1,k2
    cut: cut name: Part A=k1,k2; Part B=k3

The exam is frozen: once items are written, change them only to fix a clear
error, and record why in exam/CHANGES.md.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from modulariser.check import validate_item  # noqa: E402


def parse_block(block: str) -> dict:
    head, rest = block.split("\n", 1)
    iid, domain, kind, origin, licence, status = [x.strip() for x in head.split("|")]
    text, spec = rest.split("\n---\n", 1)
    keys, elements, edges, qs, changes, cuts = {}, [], [], [], [], []

    def g(k):
        return keys[k.strip()]

    for line in spec.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("el "):
            k, t = line[3:].split(":", 1)
            keys[k.strip()] = f"g{len(keys) + 1}"
            elements.append({"id": keys[k.strip()], "text": t.strip()})
        elif line.startswith("edges:"):
            for e in filter(None, (x.strip() for x in line[6:].split(","))):
                m = re.match(r"(\w+)\s*(>>|>)\s*(\w+)$", e)
                assert m, f"{iid}: bad edge {e}"
                a, op, b = m.groups()
                edges.append({"from": g(a), "to": g(b), "kind": "order" if op == ">>" else "uses"})
        elif line.startswith("q:"):
            q, a, needs = [x.strip() for x in line[2:].split("|")]
            qs.append({"q": q, "a": a, "needs": [g(n) for n in needs.split(",") if n.strip()]})
        elif line.startswith("ch:"):
            n, t = [x.strip() for x in line[3:].split("|")]
            changes.append({"name": n, "touches": [g(k) for k in t.split(",") if k.strip()]})
        elif line.startswith("cut:"):
            name, parts = line[4:].split(":", 1)
            cut = {"name": name.strip(), "parts": []}
            for p in parts.split(";"):
                pn, owns = p.split("=")
                cut["parts"].append({"name": pn.strip(), "owns": [g(k) for k in owns.split(",") if k.strip()]})
            cuts.append(cut)
        else:
            raise ValueError(f"{iid}: cannot parse {line!r}")
    return {
        "id": iid, "domain": domain,
        "source": {"kind": kind, "origin": origin, "licence": licence},
        "input": text.strip(), "expect_status": status,
        "elements": elements, "edges": edges, "questions": qs, "changes": changes, "cuts": cuts,
    }


def main() -> int:
    out = HERE / "items"
    out.mkdir(exist_ok=True)
    bad = 0
    n = 0
    for src in sorted((HERE / "src").glob("*.txt")):
        for block in src.read_text(encoding="utf-8").split("\n=== ")[0:]:
            block = block.strip()
            if not block:
                continue
            block = block[4:] if block.startswith("=== ") else block
            item = parse_block(block)
            errs = validate_item(item)
            if errs:
                bad += 1
                print(item["id"], errs[:3])
            (out / f"{item['id']}.json").write_text(json.dumps(item, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            n += 1
    print(f"{n} items, {bad} with errors")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
