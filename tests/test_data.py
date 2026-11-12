import random

from modulariser.check import check, validate_item
from modulariser.data.pack import load_all
from modulariser.data.synth import generate


def test_packs_parse():
    packs = load_all()
    assert len(packs) >= 30
    for p in packs.values():
        assert len(p.modules) >= 4, p.name


def test_synthetic_items_are_valid_and_targets_pass():
    packs = list(load_all().values())
    rows = generate(packs, 300, seed=3, prefix="t")
    statuses = set()
    for item, target in rows:
        assert validate_item(item) == [], item["id"]
        report = check(item, target)
        assert report["hard_fail"] == [], (item["id"], report["hard_fail"])
        assert report["status_correct"]
        if target["status"] == "ok":
            assert report["scores"]["code_score"] > 0.9
        statuses.add(target["status"])
    assert statuses == {"ok", "partial", "refuse"}


def test_every_gold_element_is_in_the_text():
    packs = list(load_all().values())
    for item, _ in generate(packs, 200, seed=5, prefix="t"):
        low = item["input"].lower()
        missing = [e["text"] for e in item["elements"] if e["text"].lower() not in low]
        # A typo may hit at most one element.
        assert len(missing) <= 1, (item["id"], missing)
