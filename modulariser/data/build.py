"""Build the synthetic train / val / test files.

    python -m modulariser.data.build --out data/ --n 4000

Test domains are held out entirely, so test scores measure transfer to
processes the model never saw. Each output line is
{"item": <gold item>, "messages": [system, user, assistant]}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..prompt import messages
from .pack import load_all
from .synth import generate

TEST_DOMAINS = {"vet clinic", "food bank", "museum exhibition", "courier depot", "airport check-in", "podcast"}


def write(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item, target in rows:
            f.write(json.dumps({"item": item, "messages": messages(item["input"], target)}, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    packs = load_all()
    train_p = [p for name, p in packs.items() if name not in TEST_DOMAINS]
    test_p = [p for name, p in packs.items() if name in TEST_DOMAINS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = generate(train_p, a.n + a.n // 20, a.seed, "syn")
    write(out / "synth_train.jsonl", rows[: a.n])
    write(out / "synth_val.jsonl", rows[a.n:])
    write(out / "synth_test.jsonl", generate(test_p, 200, a.seed + 1, "syntest"))
    print(f"train {a.n}, val {len(rows) - a.n}, test 200 ({len(train_p)} train domains, {len(test_p)} test domains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
