import json
from pathlib import Path

from modulariser.check import validate_item

ITEMS = sorted((Path(__file__).parent.parent / "exam" / "items").glob("*.json"))


def test_exam_items_valid():
    assert ITEMS
    for p in ITEMS:
        item = json.loads(p.read_text(encoding="utf-8"))
        assert validate_item(item) == [], p.name
