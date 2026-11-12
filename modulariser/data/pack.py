"""Domain packs: a small text format describing how one kind of process splits.

A pack file looks like this:

    # restaurant
    people: Ana, Raj, Mei
    ## Purchasing | Buys ingredients and checks deliveries in. | why: someone has to buy food
    a order: order ingredients
    t suppliers: supplier list
    order > suppliers          # order uses suppliers
    order >> receive           # order happens before receive
    ## cross
    purchasing.order > stock.book

`a` lines are activities (verb phrases), `t` lines are things (noun phrases).
`why:` marks a module as one that is usually needed, so when the generator drops
it the target lists it under `missing`. Module keys are the lowercased first
word of the module name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources


@dataclass
class Element:
    key: str          # module.local
    text: str
    kind: str         # "a" activity or "t" thing


@dataclass
class Module:
    key: str
    name: str
    summary: str
    why: str | None
    elements: dict[str, Element] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Pack:
    name: str
    people: list[str]
    modules: dict[str, Module]
    cross: list[tuple[str, str, str]]

    def element(self, key: str) -> Element:
        m, _ = key.split(".", 1)
        return self.modules[m].elements[key]


_EDGE = re.compile(r"^(\S+)\s*(>>|>)\s*(\S+)$")


def parse(text: str) -> Pack:
    name, people, modules, cross = "", [], {}, []
    cur: Module | None = None
    in_cross = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("#") else raw.strip()
        if not line.strip():
            continue
        if line.startswith("## "):
            head = line[3:].strip()
            if head == "cross":
                in_cross, cur = True, None
                continue
            bits = [b.strip() for b in head.split("|")]
            mname, summary = bits[0], bits[1] if len(bits) > 1 else bits[0]
            why = next((b[4:].strip() for b in bits[2:] if b.startswith("why:")), None)
            key = re.sub(r"[^a-z0-9]", "", mname.split()[0].lower())
            cur = Module(key, mname, summary, why)
            modules[key] = cur
            continue
        if line.startswith("# "):
            name = line[2:].strip()
            continue
        if line.startswith("people:"):
            people = [p.strip() for p in line[7:].split(",") if p.strip()]
            continue
        m = _EDGE.match(line.strip())
        if m:
            a, op, b = m.groups()
            kind = "order" if op == ">>" else "uses"
            if in_cross:
                cross.append((a, b, kind))
            else:
                assert cur, f"edge outside module: {line}"
                cur.edges.append((f"{cur.key}.{a}", f"{cur.key}.{b}", kind))
            continue
        km = re.match(r"^\s*([at])\s+(\w+):\s*(.+)$", line)
        if km and cur:
            kind, local, etext = km.groups()
            cur.elements[f"{cur.key}.{local}"] = Element(f"{cur.key}.{local}", etext.strip(), kind)
            continue
        raise ValueError(f"cannot parse pack line: {raw!r}")
    pack = Pack(name, people, modules, cross)
    _check(pack)
    return pack


def _check(p: Pack) -> None:
    keys = {k for m in p.modules.values() for k in m.elements}
    for m in p.modules.values():
        for a, b, _ in m.edges:
            assert a in keys and b in keys, f"{p.name}: bad edge {a} {b}"
    for a, b, kind in p.cross:
        assert a in keys and b in keys, f"{p.name}: bad cross edge {a} {b}"
        assert a.split(".")[0] != b.split(".")[0], f"{p.name}: cross edge inside module {a} {b}"


def load_all() -> dict[str, Pack]:
    out = {}
    for f in sorted(resources.files("modulariser.data.packs").iterdir()):
        if f.name.endswith(".txt"):
            pack = parse(f.read_text(encoding="utf-8"))
            out[pack.name] = pack
    return out
