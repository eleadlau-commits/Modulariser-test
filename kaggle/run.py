# Runs inside a Kaggle kernel (free T4). The GitHub workflow prepends a line
#   CONFIG = {...}
# with the task, repo ref, models and options, then pushes this file.
#
# Tasks:
#   eval   score base models and/or adapters on the synthetic test set and exam
#   train  QLoRA fine-tune one base model, save the adapter, then eval it
import json
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
    sh("pip install -q -U 'trl>=0.12' 'peft>=0.13' bitsandbytes accelerate")
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
    args = SFTConfig(
        output_dir=f"{WORK}/ckpt", num_train_epochs=CONFIG.get("epochs", 1),
        per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=CONFIG.get("lr", 2e-4),
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="no",
        fp16=True, gradient_checkpointing=True, max_length=2048, assistant_only_loss=False,
        report_to=[], optim="paged_adamw_8bit",
    )
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
    summary = {}
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


def main():
    t0 = time.time()
    log("config", json.dumps(CONFIG))
    sh("nvidia-smi", check=False)
    setup()
    task = CONFIG["task"]
    base = CONFIG["models"][0]
    adapter = os.path.join(OUT, "adapter")
    if task == "train":
        train(base, adapter)
        # Eval in a fresh process: installing vLLM swaps packages under a live torch.
        cfg = dict(CONFIG, task="eval_tuned")
        src = open(f"{WORK}/repo/kaggle/run.py").read()
        with open(f"{WORK}/eval_tuned.py", "w") as f:
            f.write(f"CONFIG = {cfg!r}\n" + src)
        r = subprocess.run([sys.executable, f"{WORK}/eval_tuned.py"])
        log("eval exit code", r.returncode)
    elif task == "eval_tuned":
        evaluate([{"base": base, "adapter": adapter, "name": "tuned"}])
    else:
        specs = [{"base": m, "adapter": None, "name": m.split("/")[-1]} for m in CONFIG["models"]]
        evaluate(specs)
    log(f"all done in {(time.time() - t0) / 60:.1f} min")


main()
