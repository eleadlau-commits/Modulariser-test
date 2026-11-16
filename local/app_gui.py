"""Modulariser desktop app: paste notes, press Cut, get the parts. Runs offline on the CPU.

Built into Modulariser.exe by .github/workflows/windows-exe.yml. The model file
(~1.9 GB) is downloaded once into %LOCALAPPDATA%\\Modulariser.
"""
import json
import os
import sys
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import messagebox, scrolledtext

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modulariser.check import CUT_SCHEMA  # noqa: E402
from modulariser.infer import cut_for, strip_meta  # noqa: E402

URL = "https://huggingface.co/elliot-elliot0lee/modulariser-test-gguf/resolve/main/model-q4_k_m.gguf"
HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Modulariser"
MODEL = HOME / "model-q4_k_m.gguf"


def render(c: dict) -> str:
    if c["status"] == "refuse":
        return f"Refused ({c['reason']})"
    el = {e["id"]: e["text"] for e in c["elements"]}
    pn = {p["id"]: p["name"] for p in c["parts"]}
    out = [f"Status: {c['status']}" + (f" ({c['reason']})" if c["reason"] else "")]
    for p in c["parts"]:
        out.append(f"\n■ {p['name']}\n  {p['summary']}")
        out += [f"    - {el[e]}" for e in p["owns"]]
    if c["links"]:
        out.append("\nLinks")
        out += [f"  - {pn[l['from']]} → {pn[l['to']]}" + (f" (uses {el[l['uses']]})" if l["uses"] else " (comes before)")
                for l in c["links"]]
    for key, title in (("missing", "Possibly missing"), ("unsure", "Unsure")):
        if c[key]:
            out.append(f"\n{title}")
            out += [f"  - {x.get('name') or pn.get(x.get('part'), '')}: {x['why']}" for x in c[key]]
    if c["unresolved"]:
        out.append("\nUnresolved")
        out += [f"  - {el[e]}" for e in c["unresolved"]]
    return "\n".join(out)


class App:
    def __init__(self, root):
        self.root, self.llm, self.grammar, self.last = root, None, None, None
        root.title("Modulariser")
        root.geometry("900x720")
        tk.Label(root, text="Paste a messy description of a process:", anchor="w").pack(fill="x", padx=10, pady=(10, 0))
        self.inp = scrolledtext.ScrolledText(root, height=12, wrap="word")
        self.inp.pack(fill="both", expand=False, padx=10, pady=5)
        bar = tk.Frame(root)
        bar.pack(fill="x", padx=10)
        self.btn = tk.Button(bar, text="Cut into parts", command=self.go)
        self.btn.pack(side="left")
        tk.Button(bar, text="Copy JSON", command=self.copy).pack(side="left", padx=8)
        self.status = tk.Label(bar, text="", anchor="w")
        self.status.pack(side="left", padx=10)
        self.out = scrolledtext.ScrolledText(root, wrap="word")
        self.out.pack(fill="both", expand=True, padx=10, pady=10)

    def say(self, s):
        self.root.after(0, lambda: self.status.config(text=s))

    def load(self):
        if self.llm:
            return
        if not MODEL.exists():
            HOME.mkdir(parents=True, exist_ok=True)
            tmp = MODEL.with_suffix(".part")

            def hook(n, block, total):
                if total > 0 and n % 100 == 0:
                    self.say(f"Downloading model (once, 1.9 GB)… {min(100, n * block * 100 // total)}%")

            urllib.request.urlretrieve(URL, tmp, hook)
            tmp.replace(MODEL)
        self.say("Loading model…")
        from llama_cpp import Llama, LlamaGrammar
        self.llm = Llama(model_path=str(MODEL), n_ctx=4096, n_threads=os.cpu_count() or 4, verbose=False)
        self.grammar = LlamaGrammar.from_json_schema(json.dumps(CUT_SCHEMA.schema), verbose=False)

    def generate(self, messages, temperature):
        r = self.llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=1800, grammar=self.grammar)
        return r["choices"][0]["message"]["content"]

    def go(self):
        text = self.inp.get("1.0", "end").strip()
        if not text:
            return
        self.btn.config(state="disabled")
        self.out.delete("1.0", "end")
        threading.Thread(target=self.work, args=(text,), daemon=True).start()

    def work(self, text):
        try:
            self.load()
            self.say("Cutting… (a minute or two)")
            cut = strip_meta(cut_for(text, self.generate))
            self.last = cut
            self.root.after(0, lambda: self.out.insert("1.0", render(cut)))
            self.say("Done.")
        except Exception as e:  # noqa: BLE001
            self.say("Error.")
            self.root.after(0, lambda: messagebox.showerror("Modulariser", str(e)))
        finally:
            self.root.after(0, lambda: self.btn.config(state="normal"))

    def copy(self):
        if self.last:
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(self.last, indent=2, ensure_ascii=False))
            self.say("JSON copied.")


def selftest(path):
    """Used by CI: load llama.cpp and build the grammar, write the outcome to a file, never show a window."""
    import traceback
    try:
        from llama_cpp import LlamaGrammar
        LlamaGrammar.from_json_schema(json.dumps(CUT_SCHEMA.schema), verbose=False)
        Path(path).write_text("selftest ok\n")
        return 0
    except BaseException:  # noqa: BLE001
        Path(path).write_text(traceback.format_exc())
        return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest(sys.argv[-1]))
    else:
        root = tk.Tk()
        App(root)
        root.mainloop()
