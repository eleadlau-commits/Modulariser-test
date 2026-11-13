"""Write a Kaggle kernel folder: run.py with CONFIG baked in, plus metadata."""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--options", default="{}")
ap.add_argument("--ref", required=True)
ap.add_argument("--run-id", required=True)
ap.add_argument("--out", default="kernel")
a = ap.parse_args()

cfg = {"task": a.task, "ref": a.ref, "models": [m.strip() for m in a.models.split(",") if m.strip()]}
cfg.update(json.loads(a.options or "{}"))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
src = Path(__file__).with_name("run.py").read_text()
(out / "run.py").write_text(f"CONFIG = {cfg!r}\n" + src)
user = os.environ["KAGGLE_USERNAME"]
slug = f"modulariser-{a.task}"  # one kernel per task; each push is a new version
meta = {
    "id": f"{user}/{slug}",
    "title": slug,
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": a.task != "export",
    "enable_tpu": False,
    "enable_internet": True,
    **({} if a.task == "export" else {"machine_shape": "NvidiaTeslaT4"}),
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [f"{os.environ['KAGGLE_USERNAME']}/{k}" for k in cfg.get("kernel_sources", [])],
    "model_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
