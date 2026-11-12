"""Code checks for a cut: hard fails first, then scores.

Hard fails gate everything: a cut that fails one gets no scores and never
reaches the judge. Scores are all in 0..1, higher is better.

Usage:
    python -m modulariser.check ITEM.json CUT.json
    python -m modulariser.check --item ITEM.json      # validate a gold item only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from importlib import resources
from itertools import combinations

from jsonschema import Draft202012Validator

from .align import align

LIMITS = {"max_parts": 12, "max_elements": 150}

# Scores averaged into code_score. The judge's blur test is scored elsewhere.
SCORE_KEYS = [
    "coverage",
    "element_precision",
    "no_ghosts",
    "relations_kept",
    "modularity_rel",
    "change_locality",
    "agreement",
]


def _schema(name: str) -> Draft202012Validator:
    text = resources.files("modulariser.schemas").joinpath(name).read_text()
    return Draft202012Validator(json.loads(text))


CUT_SCHEMA = _schema("cut.schema.json")
ITEM_SCHEMA = _schema("item.schema.json")


def _schema_errors(validator: Draft202012Validator, doc) -> list[str]:
    return [
        f"schema: {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    ]


def _dupes(ids) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


# ---------------------------------------------------------------- gold items


def validate_item(item: dict) -> list[str]:
    """Problems with a gold item. Empty list means it is usable."""
    errs = _schema_errors(ITEM_SCHEMA, item)
    if errs:
        return errs
    gids = [e["id"] for e in item["elements"]]
    known = set(gids)
    errs += [f"duplicate gold element id {d}" for d in _dupes(gids)]
    for e in item["edges"]:
        for end in (e["from"], e["to"]):
            if end not in known:
                errs.append(f"edge refers to unknown element {end}")
    for q in item["questions"]:
        errs += [f"question needs unknown element {n}" for n in q["needs"] if n not in known]
    for c in item["changes"]:
        errs += [f"change '{c['name']}' touches unknown element {t}" for t in c["touches"] if t not in known]
    if item["expect_status"] != "refuse" and not item["cuts"]:
        errs.append("an item that expects a cut needs at least one acceptable cut")
    for cut in item["cuts"]:
        owned = Counter(g for p in cut["parts"] for g in p["owns"])
        for g in sorted(known):
            if owned[g] != 1:
                errs.append(f"cut '{cut['name']}': element {g} owned {owned[g]} times (must be exactly 1)")
        errs += [f"cut '{cut['name']}': unknown element {g}" for g in owned if g not in known]
    return errs


# ---------------------------------------------------------------- hard checks


def hard_checks(cut: dict, limits: dict = LIMITS) -> list[str]:
    """Hard fails for a model cut. Any entry means the cut scores zero."""
    errs = _schema_errors(CUT_SCHEMA, cut)
    if errs:
        return errs

    eids = [e["id"] for e in cut["elements"]]
    pids = [p["id"] for p in cut["parts"]]
    errs += [f"duplicate element id {d}" for d in _dupes(eids)]
    errs += [f"duplicate part id {d}" for d in _dupes(pids)]
    known_e, known_p = set(eids), set(pids)

    if len(pids) > limits["max_parts"]:
        errs.append(f"too many parts: {len(pids)} > {limits['max_parts']}")
    if len(eids) > limits["max_elements"]:
        errs.append(f"too many elements: {len(eids)} > {limits['max_elements']}")
    if cut["status"] == "refuse":
        if cut["reason"] is None:
            errs.append("refuse needs a reason")
        return errs
    if cut["status"] == "partial" and cut["reason"] is None:
        errs.append("partial needs a reason")
    if not pids:
        errs.append(f"status {cut['status']} but no parts")

    owners: dict[str, list[str]] = defaultdict(list)
    for p in cut["parts"]:
        for e in p["owns"]:
            if e not in known_e:
                errs.append(f"part {p['id']} owns unknown element {e}")
            owners[e].append(p["id"])

    # No leaks: exactly one owner.
    for e, ps in sorted(owners.items()):
        if len(ps) > 1:
            errs.append(f"leak: element {e} owned by {', '.join(ps)}")

    unresolved = set(cut["unresolved"])
    errs += [f"unresolved refers to unknown element {e}" for e in sorted(unresolved - known_e)]
    errs += [f"element {e} is both owned and unresolved" for e in sorted(unresolved & set(owners))]
    orphans = sorted(known_e - set(owners) - unresolved)
    if orphans:
        errs.append(f"elements with no owner (own them or list as unresolved): {', '.join(orphans)}")

    for i, l in enumerate(cut["links"]):
        if l["from"] not in known_p or l["to"] not in known_p:
            errs.append(f"link {i} refers to an unknown part")
            continue
        if l["from"] == l["to"]:
            errs.append(f"link {i} links part {l['from']} to itself")
        if l["uses"] is not None and owners.get(l["uses"]) != [l["to"]]:
            errs.append(f"link {i}: {l['from']} uses {l['uses']}, which {l['to']} does not own")
    for u in cut["unsure"]:
        if u["part"] not in known_p:
            errs.append(f"unsure refers to unknown part {u['part']}")
    return errs


# ---------------------------------------------------------------- scores


def _modularity(nodes, edges, part_of) -> float:
    """Newman modularity of an undirected graph under a partition."""
    m = len(edges)
    if m == 0:
        return 0.0
    deg = Counter()
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    internal = Counter()
    degsum = Counter()
    for a, b in edges:
        if part_of[a] == part_of[b]:
            internal[part_of[a]] += 1
    for n in nodes:
        degsum[part_of[n]] += deg[n]
    return sum(internal[c] / m - (degsum[c] / (2 * m)) ** 2 for c in degsum)


def _pair_f1(a: dict, b: dict, nodes) -> float:
    """Pair-counting F1 between two partitions of the same nodes."""
    same_a = {(x, y) for x, y in combinations(sorted(nodes), 2) if a[x] == a[y]}
    same_b = {(x, y) for x, y in combinations(sorted(nodes), 2) if b[x] == b[y]}
    if not same_a and not same_b:
        return 1.0
    tp = len(same_a & same_b)
    return 2 * tp / (len(same_a) + len(same_b))


def score(item: dict, cut: dict) -> dict:
    """Scores for a cut that passed hard_checks."""
    gold = item["elements"]
    g2p = align(cut["elements"], gold)
    owner = {e: p["id"] for p in cut["parts"] for e in p["owns"]}
    # Gold element -> predicted part, for gold elements the cut owns.
    gpart = {g: owner[p] for g, p in g2p.items() if p in owner}
    covered = sorted(gpart)

    s: dict[str, float] = {}
    s["coverage"] = len(covered) / len(gold) if gold else 1.0
    s["element_precision"] = len(g2p) / len(cut["elements"]) if cut["elements"] else 0.0

    # Ghosts: parts that own nothing, or nothing any question needs.
    needed = {n for q in item["questions"] for n in q["needs"]}
    p2g = defaultdict(set)
    for g, p in gpart.items():
        p2g[p].add(g)
    ghosts = [p["id"] for p in cut["parts"] if not (p2g[p["id"]] & needed)]
    s["no_ghosts"] = 1 - len(ghosts) / len(cut["parts"])

    # Relations: each gold edge that crosses parts needs a declared link.
    linked = {frozenset((l["from"], l["to"])) for l in cut["links"]}
    crossing = [
        (a, b) for a, b in ((e["from"], e["to"]) for e in item["edges"])
        if a in gpart and b in gpart and gpart[a] != gpart[b]
    ]
    kept = [e for e in crossing if frozenset((gpart[e[0]], gpart[e[1]])) in linked]
    s["relations_kept"] = len(kept) / len(crossing) if crossing else 1.0

    # Modularity on the gold graph. The bar is the weakest acceptable cut:
    # matching any acceptable cut's modularity earns full marks.
    edges = [(e["from"], e["to"]) for e in item["edges"] if e["from"] in gpart and e["to"] in gpart]
    q_pred = _modularity(covered, edges, gpart)
    gold_parts = [{g: i for i, p in enumerate(c["parts"]) for g in p["owns"]} for c in item["cuts"]]
    all_edges = [(e["from"], e["to"]) for e in item["edges"]]
    q_gold = min((_modularity([g["id"] for g in gold], all_edges, gp) for gp in gold_parts), default=0.0)
    s["modularity_rel"] = max(0.0, min(1.0, q_pred / q_gold)) if q_gold > 0 else 1.0

    # Change locality: parts touched, against the best acceptable cut.
    loc = []
    for ch in item["changes"]:
        touched = {gpart[t] for t in ch["touches"] if t in gpart}
        best = min((len({gp[t] for t in ch["touches"]}) for gp in gold_parts), default=1)
        loc.append(best / len(touched) if touched else 0.0)
    s["change_locality"] = sum(loc) / len(loc) if loc else 1.0

    # Agreement with the closest acceptable cut, on covered elements.
    s["agreement"] = max((_pair_f1(gpart, gp, covered) for gp in gold_parts), default=0.0)

    s["code_score"] = sum(s[k] for k in SCORE_KEYS) / len(SCORE_KEYS)
    return {
        "scores": {k: round(v, 4) for k, v in s.items()},
        "details": {
            "alignment": g2p,
            "ghost_parts": ghosts,
            "lost_relations": [list(e) for e in crossing if e not in kept],
            "uncovered": sorted(set(g["id"] for g in gold) - set(covered)),
        },
    }


def check(item: dict, cut: dict, limits: dict = LIMITS) -> dict:
    """Full code report for one cut against one gold item."""
    fails = hard_checks(cut, limits)
    status = cut.get("status") if isinstance(cut, dict) else None
    report = {
        "item": item["id"],
        "hard_fail": fails,
        "status": status,
        "status_correct": status == item["expect_status"],
    }
    if fails:
        report["scores"] = {"code_score": 0.0}
    elif status == "refuse":
        # Refusing a good input is an error too: no partial credit.
        report["scores"] = {"code_score": 1.0 if item["expect_status"] == "refuse" else 0.0}
    else:
        report.update(score(item, cut))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item_pos", nargs="?", metavar="ITEM")
    ap.add_argument("cut", nargs="?", metavar="CUT")
    ap.add_argument("--item", help="validate a gold item and exit")
    a = ap.parse_args(argv)
    load = lambda p: json.load(open(p, encoding="utf-8"))
    if a.item:
        errs = validate_item(load(a.item))
        print(json.dumps({"item_ok": not errs, "errors": errs}, indent=2))
        return 1 if errs else 0
    if not (a.item_pos and a.cut):
        ap.error("give ITEM and CUT, or --item ITEM")
    item = load(a.item_pos)
    errs = validate_item(item)
    if errs:
        print(json.dumps({"item_ok": False, "errors": errs}, indent=2))
        return 2
    report = check(item, load(a.cut))
    print(json.dumps(report, indent=2))
    return 1 if report["hard_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
