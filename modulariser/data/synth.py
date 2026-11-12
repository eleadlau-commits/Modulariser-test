"""Synthetic items: sample a gold structure from a domain pack, then write a
messy description of it.

Each call to `make_item` returns (item, target):
  item   -- a gold item matching item.schema.json (used for scoring)
  target -- the cut the model should output, matching cut.schema.json

The mess is varied on purpose (style, order, noise, hints, repeats) so the
model learns to read structure, not to undo one scrambler.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from .pack import Element, Pack

# ---------------------------------------------------------------- phrasing

USES = {
    ("a", "t"): [
        "to {a}, {p} uses the {b}",
        "the {b} is what we go by when we {a}",
        "{p} checks the {b} before they {a}",
        "we {a} based on the {b}",
        "when we {a} we look at the {b}",
        "{a} needs the {b}",
        "you can't {a} without the {b}",
    ],
    ("a", "a"): [
        "how we {a} depends on how we {b}",
        "to {a} we need the result of the step where we {b}",
        "{a} relies on whoever does the job to {b}",
        "{a} needs input from: {b}",
    ],
    ("t", "t"): [
        "the {a} is built from the {b}",
        "the {a} draws on the {b}",
        "the {a} pulls its numbers from the {b}",
        "{a} is based on the {b}",
    ],
    ("t", "a"): [
        "the {a} gets updated whenever we {b}",
        "the {a} is filled in when we {b}",
        "the {a} comes out of the step where we {b}",
        "{a} depends on: {b}",
    ],
}
ORDER = [
    "first we {a}, then we {b}",
    "we always {a} before we {b}",
    "we only {b} after we {a}",
    "you have to {a} before you can {b}",
    "{b} comes after: {a}",
]
ORDER_T = ["the {a} has to be sorted before the {b}", "'{a}' comes before '{b}'"]
LONE = {
    "a": ["someone needs to {x}", "we also {x}", "and of course we {x}", "{p} is the one who tends to {x}"],
    "t": ["there's a {x}", "we keep a {x}", "{p} looks after the {x}", "there's also the {x}"],
}
PERSON = {
    "a": ["{p} is in charge when we {x}", "it's {p} who gets to {x}", "{p} handles it when we need to {x}"],
    "t": ["{p} owns the {x}", "the {x} is {p}'s job"],
}
HINT = ["on the {m} side: ", "re {m}: ", "{m} — ", "for {m}, "]
NOISE = [
    "parking is a nightmare btw",
    "(remind me to fix the printer)",
    "not sure this is all of it but here goes",
    "sorry this is a bit all over the place",
    "the coffee machine is broken again",
    "we talked about this at the last meeting too",
    "I'll tidy this up later",
    "it mostly works, honestly",
    "this has grown over the years",
    "new people always get confused by this",
    "ask me if anything doesn't make sense",
    "we might change some of this next year",
    "half of this lives in people's heads",
]
STYLES = ["notes", "email", "ticket", "transcript", "ramble", "bullets"]


def _fill(tpl: str, rng: random.Random, people: list[str], **kw) -> str:
    return tpl.format(p=rng.choice(people) if people else "someone", **kw)


def _edge_sentence(a: Element, b: Element, kind: str, rng, people) -> str:
    if kind == "order":
        tpl = rng.choice(ORDER if a.kind == b.kind == "a" else ORDER_T)
    else:
        tpl = rng.choice(USES[(a.kind, b.kind)])
    return _fill(tpl, rng, people, a=a.text, b=b.text)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _typo(s: str, rng) -> str:
    words = s.split(" ")
    i = rng.randrange(len(words))
    w = words[i]
    if len(w) > 4 and w.isalpha():
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _render(sentences: list[str], style: str, rng, people, domain: str) -> str:
    who = rng.choice(people) if people else "Sam"
    if style == "notes":
        lines = [s.lower() for s in sentences]
        sep = rng.choice(["\n", "\n- ", ". ", "; "])
        body = sep.join(lines)
        return (("- " if sep == "\n- " else "") + body).strip()
    if style == "bullets":
        return "\n".join(f"{rng.choice(['*', '-', '•'])} {_cap(s)}" for s in sentences)
    if style == "email":
        paras, cur = [], []
        for s in sentences:
            cur.append(_cap(s) + ".")
            if rng.random() < 0.3:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
        greet = rng.choice(["Hi all,", "Hey team,", "Hello,", f"Hi {who},"])
        sign = rng.choice(["Thanks,", "Cheers,", "Best,"])
        return f"{greet}\n\nHere's roughly how the {domain} side works.\n\n" + "\n\n".join(paras) + f"\n\n{sign}\n{rng.choice(people) if people else 'Jo'}"
    if style == "ticket":
        head = rng.choice([f"Document how our {domain} process works", f"Write down the {domain} workflow", "Process notes (draft)"])
        return f"Title: {head}\n\nContext:\n" + "\n".join(_cap(s) + "." for s in sentences) + "\n\nAcceptance: someone new can follow this."
    if style == "transcript":
        speakers = (people or ["A", "B"])[:3]
        out = []
        for i, s in enumerate(sentences):
            sp = speakers[i % len(speakers)] if rng.random() < 0.7 else rng.choice(speakers)
            filler = rng.choice(["", "", "yeah, ", "so ", "right, ", "um, "])
            out.append(f"{sp}: {filler}{s}.")
        return "\n".join(out)
    # ramble
    joins = [". ", ", and ", ". oh and ", ". so ", ", then ", ". also "]
    text = sentences[0]
    for s in sentences[1:]:
        text += rng.choice(joins) + s
    return _cap(text) + "."


# ---------------------------------------------------------------- structure


def _sample_structure(pack: Pack, rng: random.Random):
    mods = list(pack.modules.values())
    k = rng.randint(min(3, len(mods)), len(mods))
    keep = rng.sample(mods, k)
    dropped = [m for m in mods if m not in keep]
    elements: list[Element] = []
    part_of: dict[str, str] = {}
    for m in keep:
        els = list(m.elements.values())
        chosen = [e for e in els if rng.random() < 0.85] or [rng.choice(els)]
        for e in chosen:
            elements.append(e)
            part_of[e.key] = m.key
    present = set(part_of)
    edges = [
        (a, b, kind)
        for m in keep for (a, b, kind) in m.edges
        if a in present and b in present
    ] + [(a, b, kind) for (a, b, kind) in pack.cross if a in present and b in present]
    return keep, dropped, elements, part_of, edges


def _alt_merge(keep, edges, part_of):
    """The pair of parts with most edges between them, if any."""
    between = Counter()
    for a, b, _ in edges:
        pa, pb = part_of[a], part_of[b]
        if pa != pb:
            between[tuple(sorted((pa, pb)))] += 1
    if len(keep) < 3 or not between:
        return None, 0
    pair, n = max(between.items(), key=lambda kv: (kv[1], kv[0]))
    return pair, n


# ---------------------------------------------------------------- items


def make_item(pack: Pack, rng: random.Random, item_id: str, conflict: bool = False):
    keep, dropped, elements, part_of, edges = _sample_structure(pack, rng)
    if len(edges) < max(3, len(elements) - 3):
        return None  # too few stated relations to justify any cut
    people = pack.people
    mod_by_key = {m.key: m for m in keep}
    by_key = {e.key: e for e in elements}

    # Sentences: one per edge, plus mentions so every element appears.
    facts: list[tuple[str, set[str], str | None]] = []  # (sentence, element keys, module key)
    mentioned: set[str] = set()
    for a, b, kind in edges:
        s = _edge_sentence(by_key[a], by_key[b], kind, rng, people)
        mod = part_of[a] if part_of[a] == part_of[b] else None
        facts.append((s, {a, b}, mod))
        mentioned |= {a, b}
    for e in elements:
        if e.key not in mentioned or rng.random() < 0.15:
            tpl = rng.choice(LONE[e.kind])
            facts.append((_fill(tpl, rng, people, x=e.text), {e.key}, part_of[e.key]))
    who_qs = []
    for e in rng.sample(elements, min(len(elements), rng.randint(0, 2))):
        p = rng.choice(people) if people else "Sam"
        facts.append((rng.choice(PERSON[e.kind]).format(p=p, x=e.text), {e.key}, part_of[e.key]))
        who_qs.append({"q": f"Who is responsible for '{e.text}'?", "a": p, "needs": [e.key]})

    conflict_key = None
    if conflict:
        # An element linked to two parts, with people disagreeing where it belongs.
        cands = defaultdict(set)
        for a, b, _ in edges:
            if part_of[a] != part_of[b]:
                cands[a].add(part_of[b])
                cands[b].add(part_of[a])
        cands = {k: v for k, v in cands.items() if v}
        if not cands:
            return None
        conflict_key = rng.choice(sorted(cands))
        other = mod_by_key[rng.choice(sorted(cands[conflict_key]))]
        home = mod_by_key[part_of[conflict_key]]
        x = by_key[conflict_key].text
        facts.append((rng.choice([
            f"people disagree about who owns '{x}': some say {home.name.lower()}, some say {other.name.lower()}",
            f"honestly nobody agrees whether '{x}' is a {home.name.lower()} thing or a {other.name.lower()} thing",
            f"{x} has moved between {home.name.lower()} and {other.name.lower()} twice this year and it's still argued about",
        ]), {conflict_key}, None))

    # Order: shuffle fully, or keep module chunks together, or interleave.
    mode = rng.choice(["shuffle", "chunks", "chunks", "loose"])
    if mode == "shuffle":
        rng.shuffle(facts)
    else:
        groups = defaultdict(list)
        for f in facts:
            groups[f[2]].append(f)
        keys = list(groups)
        rng.shuffle(keys)
        facts = [f for k in keys for f in rng.sample(groups[k], len(groups[k]))]
        if mode == "loose":
            for _ in range(len(facts) // 3):
                i, j = rng.randrange(len(facts)), rng.randrange(len(facts))
                facts[i], facts[j] = facts[j], facts[i]
    hint_p = rng.choice([0.0, 0.0, 0.3])
    sentences, last_mod = [], None
    for s, _, mod in facts:
        if mod and mod != last_mod and rng.random() < hint_p:
            s = rng.choice(HINT).format(m=mod_by_key[mod].name.lower()) + s
        last_mod = mod
        sentences.append(s)
    for _ in range(rng.choice([0, 0, 1, 2])):
        sentences.insert(rng.randrange(len(sentences) + 1), rng.choice(NOISE))
    if rng.random() < 0.3:
        i = rng.randrange(len(sentences))
        sentences[i] = _typo(sentences[i], rng)
    style = rng.choice(STYLES)
    text = _render(sentences, style, rng, people, pack.name)

    # Gold ids in order of first appearance in the text.
    def first_pos(e: Element) -> int:
        i = text.lower().find(e.text.lower())
        return i if i >= 0 else len(text)

    order = sorted(elements, key=first_pos)
    gid = {e.key: f"g{i + 1}" for i, e in enumerate(order)}
    eid = {e.key: f"e{i + 1}" for i, e in enumerate(order)}

    part_order = []
    for e in order:
        if part_of[e.key] not in part_order:
            part_order.append(part_of[e.key])
    pid = {m: f"p{i + 1}" for i, m in enumerate(part_order)}

    gold_edges = [{"from": gid[a], "to": gid[b], "kind": k} for a, b, k in edges]
    canon = {"name": "by function", "parts": [
        {"name": mod_by_key[m].name, "owns": [gid[e.key] for e in order if part_of[e.key] == m]}
        for m in part_order
    ]}
    cuts = [canon]
    pair, n_between = _alt_merge(keep, edges, part_of)
    if pair:
        a_m, b_m = pair
        merged = {"name": "merged " + mod_by_key[a_m].name.lower() + " and " + mod_by_key[b_m].name.lower(), "parts": []}
        for m in part_order:
            if m == b_m:
                continue
            owns = [gid[e.key] for e in order if part_of[e.key] in ((a_m, b_m) if m == a_m else (m,))]
            merged["parts"].append({"name": mod_by_key[m].name, "owns": owns})
        cuts.append(merged)

    # Questions from edges, plus who-questions.
    qs = []
    for a, b, k in rng.sample(edges, min(len(edges), 6)):
        A, B = by_key[a].text, by_key[b].text
        if k == "order":
            qs.append({"q": f"What has to happen before '{B}'?", "a": A, "needs": [gid[a], gid[b]]})
        else:
            qs.append({"q": f"What does '{A}' depend on?", "a": B, "needs": [gid[a], gid[b]]})
    for q in who_qs:
        q["needs"] = [gid[n] for n in q["needs"]]
    qs += who_qs
    needed = {n for q in qs for n in q["needs"]}
    for e in order:  # every element should matter to at least one question
        if gid[e.key] not in needed:
            qs.append({"q": f"Is '{e.text}' part of the process?", "a": "yes", "needs": [gid[e.key]]})

    # Changes: inside one part, the element plus its in-part neighbours.
    changes = []
    for m in rng.sample(part_order, min(3, len(part_order))):
        els = [e for e in order if part_of[e.key] == m]
        e = rng.choice(els)
        nb = {e.key} | {b for a, b, _ in edges if a == e.key and part_of[b] == m} | {a for a, b, _ in edges if b == e.key and part_of[a] == m}
        verb = "change how we " if e.kind == "a" else "replace the "
        changes.append({"name": verb + e.text, "touches": sorted(gid[k] for k in nb)})

    status = "partial" if conflict_key else "ok"
    item = {
        "id": item_id,
        "domain": pack.name,
        "source": {"kind": "synthetic", "origin": f"pack:{pack.name}/{style}", "licence": "CC0"},
        "input": text,
        "expect_status": status,
        "elements": [{"id": gid[e.key], "text": e.text} for e in order],
        "edges": gold_edges,
        "questions": qs,
        "changes": changes,
        "cuts": cuts,
    }

    # Target cut.
    links, seen = [], set()
    for a, b, k in edges:
        pa, pb = part_of[a], part_of[b]
        if pa == pb or conflict_key in (a, b):
            continue
        uses = eid[b] if k == "uses" else None
        key = (pid[pa], pid[pb], uses, k)
        if key not in seen:
            seen.add(key)
            links.append({"from": pid[pa], "to": pid[pb], "uses": uses, "kind": k})
    missing = [{"name": m.name, "why": _cap(m.why)} for m in dropped if m.why]
    unsure = []
    if pair and n_between >= 2:
        a_m, b_m = pair
        sizes = {m: sum(1 for e in order if part_of[e.key] == m) for m in pair}
        small, big = sorted(pair, key=lambda m: (sizes[m], m))
        if sizes[small] <= 2:
            unsure.append({"part": pid[small], "why": f"Closely tied to {mod_by_key[big].name}; could be merged into it."})
    target = {
        "status": status,
        "reason": "conflicting" if conflict_key else None,
        "elements": [{"id": eid[e.key], "text": e.text} for e in order],
        "parts": [
            {
                "id": pid[m],
                "name": mod_by_key[m].name,
                "summary": mod_by_key[m].summary,
                "owns": [eid[e.key] for e in order if part_of[e.key] == m and e.key != conflict_key],
            }
            for m in part_order
        ],
        "links": links,
        "missing": missing,
        "unsure": unsure,
        "unresolved": [eid[conflict_key]] if conflict_key else [],
    }
    # A part left empty by the conflict is dropped, and ids renumbered.
    if any(not p["owns"] for p in target["parts"]):
        return None
    return item, target


# ---------------------------------------------------------------- refusals

SEQUENCES = [
    ["preheat the oven", "cream the butter and sugar", "beat in the eggs", "fold in the flour", "pour into the tin", "bake for 25 minutes", "leave to cool"],
    ["unbox the flat-pack", "lay out the panels", "screw the sides to the base", "fit the back panel", "attach the doors", "adjust the hinges"],
    ["wake up", "shower", "make coffee", "check the train times", "walk to the station", "catch the 7:42"],
    ["switch off the power", "remove the old socket", "connect the live wire", "connect the neutral wire", "screw the faceplate on", "turn the power back on"],
    ["boil the kettle", "warm the pot", "add the tea leaves", "pour the water", "brew for four minutes", "pour through a strainer"],
    ["jack up the car", "loosen the wheel nuts", "remove the wheel", "fit the spare", "tighten the nuts", "lower the car"],
    ["sand the wall", "fill the cracks", "sand again", "apply primer", "paint the first coat", "paint the second coat"],
    ["mix the dough", "knead for ten minutes", "leave to rise", "knock back", "shape the loaf", "prove again", "bake"],
    ["open the app", "tap new transfer", "choose the payee", "enter the amount", "confirm with your PIN", "save the receipt"],
    ["rinse the rice", "add water", "bring to the boil", "cover and simmer", "turn off the heat", "rest for ten minutes"],
]
SEQ_STYLE = [
    lambda s: "Steps: " + "; ".join(f"{i + 1}) {x}" for i, x in enumerate(s)),
    lambda s: "first " + ", then ".join(s) + ".",
    lambda s: "\n".join(f"{i + 1}. {_cap(x)}" for i, x in enumerate(s)),
    lambda s: _cap(s[0]) + ". After that " + ". Next ".join(s[1:]) + ". Done.",
]
VAGUE = [
    "we need to sort out the shop stuff, you know the usual",
    "can you make our process better? it's a mess",
    "things happen and then other things happen, the team deals with it",
    "modularise the business please",
    "it's like the old system but different",
    "stuff comes in, we do things to it, stuff goes out",
    "the process is the process, everyone knows it",
    "see attached (nothing attached)",
    "help with workflow",
    "we have some departments and they talk to each other sometimes",
]
OFF_TOPIC = [
    "Roses are red, violets are blue, the sea is so wide, and I miss you.",
    "What is the capital of Australia?",
    "Solve for x: 3x + 7 = 22.",
    "lol did you see the match last night, absolute scenes",
    "Write me a limerick about a cat who learned to fly.",
    "The weather today is mild with a chance of rain in the afternoon.",
    "Translate 'good morning' into Spanish.",
    "I think the best pizza topping is mushroom, change my mind.",
    "def add(a, b):\n    return a + b\nprint(add(2, 3))",
    "Once upon a time there was a fox who lived under a bridge and told riddles to travellers.",
]


def make_refusal(rng: random.Random, item_id: str):
    kind = rng.choice(["sequential", "sequential", "underspecified", "out_of_domain"])
    if kind == "sequential":
        seq = rng.choice(SEQUENCES)
        text = rng.choice(SEQ_STYLE)(seq)
        domain = "sequence"
    elif kind == "underspecified":
        text = rng.choice(VAGUE)
        domain = "vague"
    else:
        text = rng.choice(OFF_TOPIC)
        domain = "off-topic"
    item = {
        "id": item_id, "domain": domain,
        "source": {"kind": "synthetic", "origin": f"refusal:{kind}", "licence": "CC0"},
        "input": text, "expect_status": "refuse",
        "elements": [], "edges": [], "questions": [], "changes": [], "cuts": [],
    }
    target = {"status": "refuse", "reason": kind, "elements": [], "parts": [], "links": [],
              "missing": [], "unsure": [], "unresolved": []}
    return item, target


def generate(packs: list[Pack], n: int, seed: int, prefix: str,
             refuse_rate: float = 0.08, conflict_rate: float = 0.05):
    rng = random.Random(seed)
    out, i = [], 0
    while len(out) < n:
        i += 1
        iid = f"{prefix}-{i:05d}"
        r = rng.random()
        if r < refuse_rate:
            out.append(make_refusal(rng, iid))
            continue
        res = make_item(rng.choice(packs), rng, iid, conflict=r < refuse_rate + conflict_rate)
        if res:
            out.append(res)
    return out
