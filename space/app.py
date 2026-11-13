"""Free demo: the tuned model as a GGUF on a CPU Space, with JSON-schema
constrained decoding and the same code wrapper used in evaluation."""
import json
import os

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

from modulariser.check import CUT_SCHEMA
from modulariser.infer import cut_for, strip_meta

REPO = os.environ.get("MODEL_REPO", "")
FILE = os.environ.get("MODEL_FILE", "model-q4_k_m.gguf")
path = hf_hub_download(REPO, FILE, token=os.environ.get("HF_TOKEN"))
llm = Llama(model_path=path, n_ctx=4096, n_threads=os.cpu_count() or 2, verbose=False)
grammar = LlamaGrammar.from_json_schema(json.dumps(CUT_SCHEMA.schema))


def generate(messages, temperature):
    out = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=1800, grammar=grammar)
    return out["choices"][0]["message"]["content"]


def render(cut):
    if cut["status"] == "refuse":
        return f"**Refused** ({cut['reason']})"
    el = {e["id"]: e["text"] for e in cut["elements"]}
    pn = {p["id"]: p["name"] for p in cut["parts"]}
    md = [f"**Status:** {cut['status']}" + (f" ({cut['reason']})" if cut["reason"] else "")]
    for p in cut["parts"]:
        md.append(f"\n### {p['name']}\n{p['summary']}\n\n" + "\n".join(f"- {el[e]}" for e in p["owns"]))
    if cut["links"]:
        md.append("\n### Links")
        for l in cut["links"]:
            md.append(f"- {pn[l['from']]} → {pn[l['to']]}" + (f" (uses *{el[l['uses']]}*)" if l["uses"] else " (comes before)"))
    for key, title in (("missing", "Possibly missing"), ("unsure", "Unsure")):
        if cut[key]:
            md.append(f"\n### {title}")
            md += [f"- {x.get('name') or pn.get(x.get('part'), '')}: {x['why']}" for x in cut[key]]
    if cut["unresolved"]:
        md.append("\n### Unresolved\n" + "\n".join(f"- {el[e]}" for e in cut["unresolved"]))
    return "\n".join(md)


def run(text):
    cut = cut_for(text, generate)
    meta = cut.get("_meta", {})
    clean = strip_meta(cut)
    return render(clean), json.dumps(clean, indent=2), f"attempts: {meta.get('attempts')}"


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=14, label="Messy process description"),
    outputs=[gr.Markdown(label="Cut"), gr.Code(language="json", label="JSON"), gr.Textbox(label="Diagnostics")],
    title="Modulariser (test build)",
    description="Paste a messy description of a process. Runs on a free CPU, so expect about a minute.",
)
demo.queue(max_size=8).launch()
