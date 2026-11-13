# Runs inside a Kaggle kernel (free T4). The GitHub workflow prepends a line
#   CONFIG = {...}
# with the task, repo ref, models and options, then pushes this file.
#
# Tasks:
#   eval   score base models and/or adapters on the synthetic test set and exam
#   train  QLoRA fine-tune one base model, save the adapter, then eval it
import json
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import os
import subprocess
import sys
import time

CONFIG = globals().get("CONFIG", {"task": "eval", "ref": "main", "models": ["Qwen/Qwen2.5-3B-Instruct"]})
REPO = "https://github.com/eleadlau-commits/Modulariser-test"
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    LOG.flush()


def sh(cmd, check=True):
    log("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout:
        LOG.write(r.stdout[-4000:])
    if r.returncode and check:
        log(r.stderr[-4000:])
        raise SystemExit(f"failed: {cmd}")
    return r


def setup():
    if os.path.exists(f"{WORK}/data/synth_test.jsonl"):
        sys.path.insert(0, f"{WORK}/repo")
        return
    sh(f"git clone -q {REPO} {WORK}/repo && cd {WORK}/repo && git checkout -q {CONFIG['ref']}")
    sh(f"pip install -q -e {WORK}/repo")
    sys.path.insert(0, f"{WORK}/repo")
    sh(f"cd {WORK}/repo && python -m modulariser.data.build --out {WORK}/data --n {CONFIG.get('n_train', 4000)}")


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def exam_items():
    import glob
    items = []
    for f in sorted(glob.glob(f"{WORK}/repo/exam/items/*.json")):
        items.append(json.load(open(f, encoding="utf-8")))
    return items


# ---------------------------------------------------------------- train


def train(base, adapter_dir):
    sh("pip install -q 'trl>=0.12,<0.24' 'peft>=0.13' 'bitsandbytes>=0.45' accelerate datasets")
    sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|trl|peft|bitsandbytes|accelerate) '", check=False)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    rows = load_jsonl(f"{WORK}/data/synth_train.jsonl")
    extra = f"{WORK}/repo/data/extra_train.jsonl"
    if os.path.exists(extra):
        rows += load_jsonl(extra)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows]).shuffle(seed=0)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        torch_dtype=torch.float16, device_map={"": 0},
    )
    lora = LoraConfig(r=CONFIG.get("rank", 32), lora_alpha=CONFIG.get("rank", 32) * 2, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    import inspect
    want = dict(
        output_dir=f"{WORK}/ckpt", num_train_epochs=CONFIG.get("epochs", 1),
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=CONFIG.get("lr", 2e-4),
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="no",
        fp16=True, gradient_checkpointing=True, max_length=2048, max_seq_length=2048,
        report_to=[], optim="paged_adamw_8bit",
    )
    ok = set(inspect.signature(SFTConfig.__init__).parameters)
    args = SFTConfig(**{k: v for k, v in want.items() if k in ok})
    try:
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=lora)
    except TypeError:
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, tokenizer=tok, peft_config=lora)
    t0 = time.time()
    trainer.train()
    log(f"train done in {(time.time() - t0) / 60:.1f} min")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    hist = [h for h in trainer.state.log_history if "loss" in h]
    json.dump(hist, open(os.path.join(OUT, "train_log.json"), "w"))
    del trainer, model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- eval


def evaluate(models):
    """models: list of {"base": hf id, "adapter": path or None, "name": label}."""
    sh("pip install -q vllm", check=True)
    from modulariser.check import CUT_SCHEMA, check
    from modulariser.infer import cut_for, strip_meta

    synth = [r["item"] for r in load_jsonl(f"{WORK}/data/synth_test.jsonl")][: CONFIG.get("n_test", 100)]
    exam = exam_items()
    sets = {"synth_test": synth, "exam": exam}
    sp = os.path.join(OUT, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    for spec in models:
        res = eval_one(spec, sets, CUT_SCHEMA.schema, check, cut_for, strip_meta)
        summary[spec["name"]] = res
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        log(spec["name"], json.dumps(res))
    return summary


def eval_one(spec, sets, schema, check, cut_for, strip_meta):
    import gc

    import torch
    from vllm import LLM, SamplingParams

    kw = dict(model=spec["base"], dtype="half", max_model_len=4096, gpu_memory_utilization=0.9,
              enforce_eager=True, tensor_parallel_size=spec.get("tp", 1))
    if spec.get("adapter"):
        kw.update(enable_lora=True, max_lora_rank=64)
    llm = LLM(**kw)
    lora = None
    if spec.get("adapter"):
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("a", 1, spec["adapter"])

    def params(temp):
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=temp, max_tokens=1800, structured_outputs=StructuredOutputsParams(json=schema))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=temp, max_tokens=1800, guided_decoding=GuidedDecodingParams(json=schema))

    # Batch the first attempt for speed; the wrapper's retry goes one by one.
    out = {}
    for set_name, items in sets.items():
        if not items:
            continue
        from modulariser.prompt import messages
        convs = [messages(it["input"]) for it in items]
        t0 = time.time()
        first = llm.chat(convs, params(0.0), lora_request=lora, use_tqdm=False)
        cache = {i: o.outputs[0].text for i, o in enumerate(first)}
        rows = []
        for i, it in enumerate(items):
            calls = {"n": 0}

            def gen(msgs, temp, i=i):
                calls["n"] += 1
                if calls["n"] == 1:
                    return cache[i]
                r = llm.chat([msgs], params(temp), lora_request=lora, use_tqdm=False)
                return r[0].outputs[0].text

            cut = cut_for(it["input"], gen)
            rep = check(it, strip_meta(cut))
            rows.append({"id": it["id"], "domain": it["domain"], "expect": it["expect_status"],
                         "status": cut["status"], "meta": cut["_meta"], "scores": rep["scores"],
                         "hard_fail": rep["hard_fail"], "raw_first": cache[i][:3000], "cut": strip_meta(cut)})
        secs = time.time() - t0
        with open(os.path.join(OUT, f"{spec['name']}__{set_name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        ok_items = [r for r in rows if r["expect"] != "refuse"]
        out[set_name] = {
            "n": n,
            "code_score": round(sum(r["scores"]["code_score"] for r in rows) / n, 4),
            "code_score_cut_items": round(sum(r["scores"]["code_score"] for r in ok_items) / max(1, len(ok_items)), 4),
            "status_acc": round(sum(r["status"] == r["expect"] for r in rows) / n, 4),
            "first_try_pass": round(sum(r["meta"]["attempts"] == 1 for r in rows) / n, 4),
            "wrapper_refused": sum(bool(r["meta"].get("wrapper_refused")) for r in rows),
            "secs_per_item": round(secs / n, 2),
        }
        for k in ["coverage", "element_precision", "no_ghosts", "relations_kept", "modularity_rel", "change_locality", "agreement"]:
            vals = [r["scores"][k] for r in ok_items if k in r["scores"]]
            if vals:
                out[set_name][k] = round(sum(vals) / len(vals), 4)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- judge

JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
JUDGE_PROMPT = """You are testing whether a summary of a process kept what matters.
Below is a blurred view of a process: only its parts, what each part owns, and the links between parts.
Using ONLY this view (no outside knowledge about this particular process), could someone answer the question with the expected answer?
Reply with exactly one word: yes or no.

VIEW:
{view}

QUESTION: {q}
EXPECTED ANSWER: {a}"""


def blur_view(cut):
    el = {e["id"]: e["text"] for e in cut.get("elements", [])}
    pn = {p["id"]: p["name"] for p in cut.get("parts", [])}
    lines = []
    for p in cut.get("parts", []):
        owns = "; ".join(el.get(e, e) for e in p["owns"])
        lines.append(f"- {p['name']}: {p['summary']} Owns: {owns}")
    for l in cut.get("links", []):
        a, b = pn.get(l["from"], l["from"]), pn.get(l["to"], l["to"])
        if l["kind"] == "order":
            lines.append(f"- {a} must happen before {b}")
        else:
            lines.append(f"- {a} uses '{el.get(l['uses'], l['uses'])}' from {b}")
    if cut.get("unresolved"):
        lines.append("- Unresolved: " + "; ".join(el.get(e, e) for e in cut["unresolved"]))
    return "\n".join(lines)


def judge():
    """Blur test on every result file in OUT; adds blur_score to summary.json."""
    import glob

    sh("pip install -q vllm", check=True)
    from vllm import LLM, SamplingParams

    items = {r["item"]["id"]: r["item"] for r in load_jsonl(f"{WORK}/data/synth_test.jsonl")}
    items.update({it["id"]: it for it in exam_items()})
    llm = LLM(model=CONFIG.get("judge_model", JUDGE_MODEL), dtype="half", max_model_len=4096,
              gpu_memory_utilization=0.9, enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=3)
    summary = json.load(open(os.path.join(OUT, "summary.json")))
    for f in sorted(glob.glob(os.path.join(OUT, "*__*.jsonl"))):
        name, set_name = os.path.basename(f)[:-6].split("__")
        rows = load_jsonl(f)
        prompts, index = [], []
        for ri, r in enumerate(rows):
            it = items.get(r["id"])
            if not it or it["expect_status"] == "refuse" or r["status"] == "refuse" or r["hard_fail"]:
                continue
            view = blur_view(r["cut"])
            for q in it["questions"]:
                prompts.append([{"role": "user", "content": JUDGE_PROMPT.format(view=view, q=q["q"], a=q["a"])}])
                index.append(ri)
        outs = llm.chat(prompts, sp, use_tqdm=False) if prompts else []
        per = {}
        for ri, o in zip(index, outs):
            per.setdefault(ri, []).append(o.outputs[0].text.strip().lower().startswith("yes"))
        cut_items = [r for r in rows if items.get(r["id"]) and items[r["id"]]["expect_status"] != "refuse"]
        for ri, r in enumerate(rows):
            r["blur_score"] = sum(per[ri]) / len(per[ri]) if ri in per else 0.0
        with open(f, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        blur = sum(r["blur_score"] for r in cut_items) / max(1, len(cut_items))
        summary.setdefault(name, {}).setdefault(set_name, {})["blur_score"] = round(blur, 4)
        # Final score per the design: code checks and blur test, equal weight, on cut items.
        cs = summary[name][set_name].get("code_score_cut_items", 0)
        summary[name][set_name]["final_score"] = round((cs + blur) / 2, 4)
        log(name, set_name, "blur", round(blur, 4))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


# ---------------------------------------------------------------- teacher (real data)

TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
MESSY_PROMPT = """Below is official guidance describing a process. Rewrite it as the messy, informal notes a staff member might jot down for a new colleague: out of order in places, chatty, with some abbreviations, maybe a side remark. Keep every activity, document, system and dependency that matters. Do not add new steps. Output only the notes.

GUIDANCE:
{text}"""


def fetch_govuk(n):
    """Process-like GOV.UK guides (Open Government Licence v3)."""
    import html
    import re
    import urllib.request

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "modulariser-research"})
        return json.load(urllib.request.urlopen(req, timeout=30))

    links, start = [], 0
    while len(links) < n * 2 and start < 4000:
        res = get(f"https://www.gov.uk/api/search.json?filter_format=guide&count=500&start={start}&fields=link,title")
        if not res.get("results"):
            break
        links += [r["link"] for r in res["results"]]
        start += 500
    docs = []
    for link in links:
        if len(docs) >= n:
            break
        try:
            c = get("https://www.gov.uk/api/content" + link)
        except Exception:
            continue
        parts = c.get("details", {}).get("parts") or []
        body = " ".join(p.get("body", "") for p in parts) or c.get("details", {}).get("body", "")
        text = html.unescape(re.sub(r"<[^>]+>", " ", body))
        text = re.sub(r"\s+", " ", text).strip()
        if 800 <= len(text):
            docs.append({"url": "https://www.gov.uk" + link, "title": c.get("title", ""), "text": text[:3500]})
    log("fetched", len(docs), "gov.uk guides")
    return docs


def teacher():
    sh("pip install -q vllm", check=True)
    from vllm import LLM, SamplingParams
    from modulariser.check import CUT_SCHEMA, _pair_f1, hard_checks
    from modulariser.align import align
    from modulariser.prompt import messages

    docs = fetch_govuk(CONFIG.get("n_docs", 1200))
    llm = LLM(model=CONFIG.get("teacher_model", TEACHER_MODEL), dtype="half", max_model_len=6144,
              gpu_memory_utilization=0.92, enforce_eager=True, tensor_parallel_size=CONFIG.get("tp", 2))
    plain = SamplingParams(temperature=0.8, max_tokens=1200)

    def guided(temp):
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=temp, max_tokens=2000, structured_outputs=StructuredOutputsParams(json=CUT_SCHEMA.schema))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=temp, max_tokens=2000, guided_decoding=GuidedDecodingParams(json=CUT_SCHEMA.schema))

    out_path = os.path.join(OUT, "real_train.jsonl")
    kept = 0
    deadline = time.time() + CONFIG.get("hours", 7) * 3600
    for i in range(0, len(docs), 100):
        if time.time() > deadline:
            log("time budget reached")
            break
        batch = docs[i:i + 100]
        messy = [o.outputs[0].text.strip() for o in llm.chat(
            [[{"role": "user", "content": MESSY_PROMPT.format(text=d["text"])}] for d in batch], plain, use_tqdm=False)]
        # The cut is made from the clean guidance (more reliable), twice, and kept only if both agree.
        convs = [messages(d["text"]) for d in batch]
        a = llm.chat(convs, guided(0.0), use_tqdm=False)
        b = llm.chat(convs, guided(0.7), use_tqdm=False)
        with open(out_path, "a") as f:
            for d, m, oa, ob in zip(batch, messy, a, b):
                try:
                    ca, cb = json.loads(oa.outputs[0].text), json.loads(ob.outputs[0].text)
                except Exception:
                    continue
                if hard_checks(ca) or hard_checks(cb) or ca["status"] != cb["status"]:
                    continue
                if ca["status"] == "ok":
                    if not (2 <= len(ca["parts"]) <= 9 and 5 <= len(ca["elements"]) <= 40 and ca["links"]):
                        continue
                    g = [{"id": e["id"].replace("e", "g"), "text": e["text"]} for e in ca["elements"]]
                    m_ = align(cb["elements"], g)
                    pa = {e.replace("e", "g"): p["id"] for p in ca["parts"] for e in p["owns"]}
                    pb = {e: p["id"] for p in cb["parts"] for e in p["owns"]}
                    common = [gid for gid, eid in m_.items() if gid in pa and eid in pb]
                    if len(common) < 0.7 * len(g):
                        continue
                    agree = _pair_f1({k: pa[k] for k in common}, {k: pb[m_[k]] for k in common}, common)
                    if agree < CONFIG.get("min_agree", 0.5):
                        continue
                    # The messy rewrite must still mention most elements.
                    low = m.lower()
                    hits = sum(any(w in low for w in e["text"].lower().split() if len(w) > 3) for e in ca["elements"])
                    if hits < 0.8 * len(ca["elements"]):
                        continue
                item_id = f"govuk-{i + batch.index(d):05d}"
                f.write(json.dumps({"id": item_id, "source": {"kind": "real", "origin": d["url"], "licence": "OGL-UK-3.0"},
                                    "domain": "govuk:" + d["title"][:60], "messages": messages(m, ca)}) + "\n")
                kept += 1
        log(f"teacher: {i + len(batch)} docs seen, {kept} kept")


# ---------------------------------------------------------------- export (GGUF for the demo)


def export():
    """Merge the adapter from the train kernel into the base and write a Q4_K_M GGUF."""
    import glob

    src = CONFIG.get("adapter_dir") or next(iter(glob.glob("/kaggle/input/*/out/adapter")), None)
    if not src:
        raise SystemExit("no adapter found under /kaggle/input")
    log("adapter", src)
    sh("pip uninstall -y -q torchao", check=False)  # an old torchao breaks peft's import
    sh("pip install -q sentencepiece gguf")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = CONFIG["models"][0]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, src).merge_and_unload()
    merged = f"{WORK}/merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    del model
    sh(f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    sh(f"pip install -q -r {WORK}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt", check=False)
    sh(f"python {WORK}/llama.cpp/convert_hf_to_gguf.py {merged} --outtype f16 --outfile {WORK}/model-f16.gguf")
    sh(f"cd {WORK}/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null && cmake --build build --target llama-quantize -j 4 > /dev/null")
    sh(f"{WORK}/llama.cpp/build/bin/llama-quantize {WORK}/model-f16.gguf {OUT}/model-q4_k_m.gguf Q4_K_M")
    sh(f"rm -rf {merged} {WORK}/model-f16.gguf {WORK}/llama.cpp", check=False)
    log("gguf size MB", os.path.getsize(f"{OUT}/model-q4_k_m.gguf") // 2**20)


def subrun(task):
    """Run this script again as a fresh process with another task."""
    cfg = dict(CONFIG, task=task)
    src = open(f"{WORK}/repo/kaggle/run.py").read()
    path = f"{WORK}/{task}.py"
    with open(path, "w") as f:
        f.write(f"CONFIG = {cfg!r}\n" + src)
    r = subprocess.run([sys.executable, path])
    log(task, "exit code", r.returncode)


def main():
    t0 = time.time()
    log("config", json.dumps(CONFIG))
    task = CONFIG["task"]
    if task in ("train", "eval", "teacher", "evaladapter"):
        sh("nvidia-smi", check=False)
    setup()
    base = CONFIG["models"][0]
    adapter = os.path.join(OUT, "adapter")
    if task == "train":
        # Fresh processes for every GPU stage: a parent that has touched CUDA keeps
        # memory that vLLM then cannot get, and installing vLLM swaps packages.
        subrun("train_only")
        subrun("eval_tuned")
    elif task == "train_only":
        train(base, adapter)
    elif task == "evaladapter":
        import glob
        src = next(iter(glob.glob("/kaggle/input/*/out/adapter") + glob.glob("/kaggle/input/**/adapter", recursive=True)), None)
        log("adapter from", src)
        evaluate([{"base": base, "adapter": src, "name": "tuned"}])
    elif task == "eval_tuned":
        evaluate([{"base": base, "adapter": adapter, "name": "tuned"}])
    elif task == "eval_one":
        m = CONFIG["one"]
        evaluate([{"base": m, "adapter": None, "name": m.split("/")[-1]}])
    elif task == "judge":
        judge()
    elif task == "teacher":
        teacher()
    elif task == "export":
        export()
    else:
        for m in CONFIG["models"]:
            subrun_cfg = dict(CONFIG, one=m)
            CONFIG.update(subrun_cfg)
            subrun("eval_one")
    if task in ("train", "eval", "evaladapter") and CONFIG.get("judge", True):
        subrun("judge")
    log(f"all done in {(time.time() - t0) / 60:.1f} min")
    if task in ("train", "eval", "teacher", "evaladapter", "export"):
        sh(f"rm -rf {WORK}/repo {WORK}/data {WORK}/ckpt", check=False)  # keep the output small


try:
    main()
except BaseException:
    import traceback
    log("CRASH", traceback.format_exc())
    raise
