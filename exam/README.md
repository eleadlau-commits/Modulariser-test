# Frozen exam (60 items)

Written once, before any model was tested, and then frozen. Edit an item only
to fix a clear error, and log the reason in `CHANGES.md`.

- 47 items expect a cut, 6 expect `partial` (an element whose owner is disputed), 7 expect a refusal.
- 22 are paraphrased from real public process documents (`source.kind = real`, the origin URL and the source licence are recorded); 38 are hand-written.
- None of the domains appear in the training packs.
- Sources are in `src/*.txt` in a compact format. `python exam/build_exam.py` writes `items/*.json` and validates them.
