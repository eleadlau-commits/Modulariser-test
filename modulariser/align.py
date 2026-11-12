"""Match the elements a model extracted to the gold elements.

The model never sees gold ids, so its elements must be aligned by text before
any graph score means anything. Greedy one-to-one matching on a simple text
similarity is enough for v1; swap in embeddings later if it proves too crude.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_STOP = {
    "a", "an", "the", "of", "to", "and", "or", "for", "in", "on", "at", "by",
    "with", "from", "is", "are", "be", "it", "its", "this", "that", "into",
}
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Best of word overlap (Jaccard) and character similarity, in 0..1."""
    ta, tb = _tokens(a), _tokens(b)
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    seq = SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
    return max(jac, seq)


def align(pred: list[dict], gold: list[dict], threshold: float = 0.5) -> dict[str, str]:
    """Return a one-to-one map from gold element id to predicted element id."""
    pairs = []
    for g in gold:
        for p in pred:
            s = similarity(g["text"], p["text"])
            if s >= threshold:
                pairs.append((s, g["id"], p["id"]))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    g2p: dict[str, str] = {}
    used: set[str] = set()
    for _, gid, pid in pairs:
        if gid not in g2p and pid not in used:
            g2p[gid] = pid
            used.add(pid)
    return g2p
