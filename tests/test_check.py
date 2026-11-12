import copy
import json
from pathlib import Path

import pytest

from modulariser.check import check, hard_checks, validate_item

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def item():
    return json.loads((EX / "bakery.item.json").read_text(encoding="utf-8"))


@pytest.fixture
def cut():
    return json.loads((EX / "bakery.cut.json").read_text(encoding="utf-8"))


def test_example_item_is_valid(item):
    assert validate_item(item) == []


def test_good_cut_passes_and_scores_high(item, cut):
    r = check(item, cut)
    assert r["hard_fail"] == []
    assert r["scores"]["coverage"] == 1.0
    assert r["scores"]["agreement"] == 1.0
    assert r["scores"]["code_score"] > 0.95


def test_leak_is_hard_fail(item, cut):
    cut["parts"][1]["owns"].append("e1")  # Stock also owns the supplier list
    r = check(item, cut)
    assert any("leak" in f for f in r["hard_fail"])
    assert r["scores"]["code_score"] == 0.0


def test_orphan_element_is_hard_fail(cut):
    cut["parts"][3]["owns"].remove("e9")
    assert any("no owner" in f for f in hard_checks(cut))


def test_orphan_allowed_when_unresolved(cut):
    cut["parts"][3]["owns"].remove("e9")
    cut["unresolved"] = ["e9"]
    cut["status"], cut["reason"] = "partial", "underspecified"
    assert hard_checks(cut) == []


def test_link_must_use_element_owned_by_target(cut):
    cut["links"][0]["uses"] = "e5"  # p2 does not own e5
    assert any("does not own" in f for f in hard_checks(cut))


def test_schema_error_is_hard_fail(cut):
    del cut["parts"][0]["summary"]
    assert any(f.startswith("schema") for f in hard_checks(cut))


def test_missing_link_loses_relation(item, cut):
    cut["links"] = cut["links"][:1]  # drop Production -> Stock
    r = check(item, cut)
    assert r["scores"]["relations_kept"] < 1.0
    assert ["g5", "g4"] in r["details"]["lost_relations"]


def test_ghost_part_is_counted(item, cut):
    cut["elements"].append({"id": "e10", "text": "staff birthday calendar"})
    cut["parts"].append({"id": "p5", "name": "Social", "summary": "Birthdays.", "owns": ["e10"]})
    r = check(item, cut)
    assert r["details"]["ghost_parts"] == ["p5"]
    assert r["scores"]["no_ghosts"] == 0.8


def test_one_big_part_scores_low(item, cut):
    everything = [e["id"] for e in cut["elements"]]
    cut["parts"] = [{"id": "p1", "name": "Bakery", "summary": "Everything.", "owns": everything}]
    cut["links"], cut["unsure"] = [], []
    r = check(item, cut)
    assert r["hard_fail"] == []
    # One part is trivially "local" under change; modularity catches it.
    assert r["scores"]["modularity_rel"] == 0.0
    assert r["scores"]["agreement"] < 0.6
    assert r["scores"]["code_score"] < 0.8


def test_wrong_refusal_scores_zero(item):
    refusal = {"status": "refuse", "reason": "sequential", "elements": [], "parts": [],
               "links": [], "missing": [], "unsure": [], "unresolved": []}
    r = check(item, refusal)
    assert r["hard_fail"] == []
    assert r["status_correct"] is False
    assert r["scores"]["code_score"] == 0.0


def test_right_refusal_scores_one(item):
    item = copy.deepcopy(item)
    item["expect_status"] = "refuse"
    refusal = {"status": "refuse", "reason": "sequential", "elements": [], "parts": [],
               "links": [], "missing": [], "unsure": [], "unresolved": []}
    assert check(item, refusal)["scores"]["code_score"] == 1.0


def test_bad_gold_cut_is_rejected(item):
    item["cuts"][0]["parts"][0]["owns"].append("g4")  # g4 now owned twice
    assert any("owned 2 times" in e for e in validate_item(item))
