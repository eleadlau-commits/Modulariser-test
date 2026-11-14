"""Run the modulariser offline on your own computer (CPU only, no GPU needed).

Same wrapper as the exam: JSON-schema grammar, hard checks, repair, one retry.

    python local/run_local.py notes.txt          # cut a text file
    python local/run_local.py                    # paste text, end with Ctrl+Z then Enter (Windows) or Ctrl+D

The model file is downloaded once (about 1.9 GB) into the models folder next to
this script; after that no internet is needed. Use --model to point at a GGUF
you already have.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulariser.check import CUT_SCHEMA  # noqa: E402
from modulariser.infer import cut_for, strip_meta  # noqa: E402

URL = "https://huggingface.co/elliot-elliot0lee/modulariser-test-gguf/resolve/main/model-q4_k_m.gguf"
DEFAULT = Path(__file__).resolve().parent / "models" / "model-q4_k_m.gguf"


def fetch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading the model once (~1.9 GB) to {path} ...", file=sys.stderr)

    def hook(n, block, total):
        if total > 0 and n % 200 == 0:
            print(f"\r  {min(100, n * block * 100 // total)}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(URL, tmp, hook)
    tmp.replace(path)
    print("\n  done", file=sys.stderr)


def render(c: dict) -> str:
    if c["status"] == "refuse":
        return f"Refused ({c['reason']})"
    el = {e["id"]: e["text"] for e in c["elements"]}
    pn = {p["id"]: p["name"] for p in c["parts"]}
    out = [f"Status: {c['status']}" + (f" ({c['reason']})" if c["reason"] else "")]
    for p in c["parts"]:
        out.append(f"\n## {p['name']}\n{p['summary']}")
        out += [f"  - {el[e]}" for e in p["owns"]]
    if c["links"]:
        out.append("\n## Links")
        out += [f"  - {pn[l['from']]} -> {pn[l['to']]}" + (f" (uses {el[l['uses']]})" if l["uses"] else " (comes before)")
                for l in c["links"]]
    for key, title in (("missing", "Possibly missing"), ("unsure", "Unsure")):
        if c[key]:
            out.append(f"\n## {title}")
            out += [f"  - {x.get('name') or pn.get(x.get('part'), '')}: {x['why']}" for x in c[key]]
    if c["unresolved"]:
        out.append("\n## Unresolved")
        out += [f"  - {el[e]}" for e in c["unresolved"]]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="text file to cut (default: read from the keyboard)")
    ap.add_argument("--model", default=str(DEFAULT), help="path to the GGUF model")
    ap.add_argument("--json", action="store_true", help="print only the JSON")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    model = Path(a.model)
    if not model.exists():
        fetch(model)
    text = Path(a.file).read_text(encoding="utf-8") if a.file else sys.stdin.read()

    from llama_cpp import Llama, LlamaGrammar
    llm = Llama(model_path=str(model), n_ctx=4096, n_threads=a.threads, verbose=False)
    grammar = LlamaGrammar.from_json_schema(json.dumps(CUT_SCHEMA.schema), verbose=False)

    def generate(messages, temperature):
        r = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=1800, grammar=grammar)
        return r["choices"][0]["message"]["content"]

    print("Cutting (a minute or two on a laptop CPU) ...", file=sys.stderr)
    cut = cut_for(text, generate)
    meta = cut.get("_meta", {})
    clean = strip_meta(cut)
    if a.json:
        print(json.dumps(clean, indent=2, ensure_ascii=False))
        return
    print(render(clean))
    print("\n--- JSON ---\n" + json.dumps(clean, indent=2, ensure_ascii=False))
    print(f"\n(attempts: {meta.get('attempts')}, repaired: {meta.get('repaired', False)})", file=sys.stderr)


if __name__ == "__main__":
    main()
