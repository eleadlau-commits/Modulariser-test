# Modulariser

A small fine-tuned language model that takes a messy description of a process (notes, a plan, a procedure, a codebase outline) and proposes a cut into well-fitting parts. It says what each part owns, lists the links between parts, and names parts it thinks are missing or cuts it is unsure of. Its output is strict JSON.

**Theory.** An abstraction is a blur from above: you stop seeing a difference while the questions you care about stay answerable. A good cut loses nothing needed (judged against a question set Q), keeps the relations that mattered below as links between parts, has no ghosts (parts that answer nothing), and has no leaks (every element has exactly one owner).

The full design (scoring, data, model, feedback loop, limits) is in the [Modulariser design doc](https://claude.ai/artifact/49cqgpJ66oka12bBV17hFN).

## What's here

| Path | What it is |
| --- | --- |
| `modulariser/schemas/cut.schema.json` | The model's output format, frozen before any data is made |
| `modulariser/schemas/item.schema.json` | A gold item: input, gold elements and edges, Q, change scenarios, acceptable cuts |
| `modulariser/check.py` | Code checks: hard fails, then scores |
| `modulariser/align.py` | Matches the model's elements to gold elements by text |
| `examples/` | One hand-written bakery item and a good cut for it |

## Install and run

Plain Python, works on Windows. No GPU needed for this part.

```
pip install -e .[dev]
python -m modulariser.check --item examples/bakery.item.json
python -m modulariser.check examples/bakery.item.json examples/bakery.cut.json
python -m pytest
```

## The checks

**Hard fails** give a score of 0 and the cut never reaches the judge:

- the output doesn't match the schema, or has too many parts or elements;
- a leak: an element owned by more than one part;
- an element with no owner (a partial cut must list it under `unresolved`);
- a link whose `uses` element is not owned by the part it points to;
- unknown or duplicate ids, or a refusal or partial cut with no reason.

**Scores**, all 0 to 1, are averaged into `code_score`:

| Score | Meaning |
| --- | --- |
| `coverage` | Share of gold elements the cut owns |
| `element_precision` | Share of the cut's elements that match a gold element (penalises invented ones) |
| `no_ghosts` | Share of parts that own something a question in Q needs |
| `relations_kept` | Share of gold edges crossing parts that have a declared link |
| `modularity_rel` | Modularity on the gold graph, against the weakest acceptable cut |
| `change_locality` | Parts touched by each scripted change, against the best acceptable cut |
| `agreement` | Pair-counting F1 against the closest acceptable cut |

A refusal scores 1 if the item expects one, and 0 otherwise.

## Known limits of v1

- Element alignment is plain text similarity. It will miss paraphrases, so swap in embeddings once real data shows how often it fails.
- `change_locality` alone rewards one giant part. `modularity_rel` and `agreement` catch it, but the equal-weight average is generous. Weights should be set once the judge is calibrated.
- Link `kind` (uses or order) is not scored yet.
- The judge's blur test is the main score and is not built yet.

## Next steps

1. Generate about 60 exam items and their Q with the cloud model, then prune them by hand.
2. Calibrate the judge against your own ratings on 20 items.
3. Test the candidate models untrained on the exam and choose the size.
