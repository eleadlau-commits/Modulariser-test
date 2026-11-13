import copy
import json
from pathlib import Path

from modulariser.check import hard_checks
from modulariser.infer import cut_for, repair

EX = Path(__file__).parent.parent / "examples"
GOOD = json.loads((EX / "bakery.cut.json").read_text())


def test_repair_leaves_good_cut_valid():
    assert hard_checks(repair(GOOD)) == []


def test_repair_fixes_leak_orphan_and_bad_link():
    bad = copy.deepcopy(GOOD)
    bad["parts"][0]["owns"].append("e4")          # leak: e4 also in p2
    bad["parts"][2]["owns"].remove("e6")          # orphan e6
    bad["links"].append({"from": "p4", "to": "p1", "uses": "e5", "kind": "uses"})  # e5 is p3's
    assert hard_checks(bad)
    fixed = repair(bad)
    assert hard_checks(fixed) == []
    owner = {e: p["id"] for p in fixed["parts"] for e in p["owns"]}
    assert owner["e4"] == "p1" and owner["e6"] == "p3"
    assert {"from": "p4", "to": "p3", "uses": "e5", "kind": "uses"} in fixed["links"]


def test_wrapper_marks_repairs():
    bad = copy.deepcopy(GOOD)
    bad["parts"][2]["owns"].remove("e6")
    out = cut_for("some text", lambda m, t: json.dumps(bad))
    assert out["_meta"]["repaired"] is True and out["status"] == "ok"
    out = cut_for("some text", lambda m, t: json.dumps(GOOD))
    assert out["_meta"]["repaired"] is False
