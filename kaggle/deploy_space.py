"""Upload the GGUF (split into <512 MB chunks) to a public model repo and deploy
the free static Space, which runs the model in the visitor's browser.

Gradio/Docker Spaces on free CPU now need a paid plan; static Spaces stay free.
"""
import argparse
import glob
import io
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def split(gguf: str, out: str) -> list[str]:
    """Split with llama.cpp's gguf-split from its latest Linux release; [] if unavailable."""
    try:
        rel = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"))
        a = next(a for a in rel["assets"] if "bin-ubuntu-x64" in a["name"] and a["name"].endswith((".zip", ".tar.gz")))
        data = io.BytesIO(urllib.request.urlopen(a["browser_download_url"]).read())
        if a["name"].endswith(".zip"):
            zipfile.ZipFile(data).extractall(out + "/bin")
        else:
            tarfile.open(fileobj=data, mode="r:gz").extractall(out + "/bin", filter="data")
        exe = next(p for p in glob.glob(out + "/bin/**/llama-gguf-split", recursive=True))
        os.chmod(exe, 0o755)
        env = {**os.environ, "LD_LIBRARY_PATH": str(Path(exe).parent)}
        subprocess.run([exe, "--split-max-size", "480M", gguf, out + "/model-q4_k_m"], check=True, env=env)
        return sorted(glob.glob(out + "/model-q4_k_m-*-of-*.gguf"))
    except Exception as e:  # noqa: BLE001
        print("split failed, uploading one file:", repr(e))
        return []


def config_js() -> str:
    import sys
    sys.path.insert(0, str(ROOT))
    from modulariser.check import CUT_SCHEMA
    from modulariser.prompt import MAX_INPUT_CHARS, SYSTEM
    return {"system": SYSTEM, "schema": CUT_SCHEMA.schema, "maxChars": MAX_INPUT_CHARS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    model_repo, space = f"{user}/modulariser-test-gguf", f"{user}/modulariser-test"
    api.create_repo(model_repo, private=False, exist_ok=True)
    api.update_repo_settings(model_repo, private=False)
    tmp = tempfile.mkdtemp()
    chunks = split(a.gguf, tmp)
    if chunks:
        old = [f for f in api.list_repo_files(model_repo) if f.endswith(".gguf")]
        for f in old:
            api.delete_file(f, model_repo, commit_message="replace model")
        for c in chunks:
            api.upload_file(path_or_fileobj=c, path_in_repo=Path(c).name, repo_id=model_repo,
                            commit_message=f"GGUF chunk from run {a.run_id}")
        first = Path(chunks[0]).name
    else:
        api.upload_file(path_or_fileobj=a.gguf, path_in_repo="model-q4_k_m.gguf", repo_id=model_repo,
                        commit_message=f"GGUF from run {a.run_id}")
        first = "model-q4_k_m.gguf"
    cfg = {"repo": model_repo, "file": first, "run": a.run_id, **config_js()}
    site = Path(tmp) / "site"
    site.mkdir()
    for f in ("index.html", "README.md"):
        (site / f).write_text((ROOT / "space" / f).read_text())
    (site / "config.js").write_text("export const CONFIG = " + json.dumps(cfg, ensure_ascii=False) + ";\n")
    api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    for f in api.list_repo_files(space, repo_type="space"):
        if f in ("app.py", "requirements.txt"):
            api.delete_file(f, space, repo_type="space", commit_message="static Space now")
    api.upload_folder(folder_path=str(site), repo_id=space, repo_type="space", commit_message=f"deploy run {a.run_id}")
    print("space:", f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
