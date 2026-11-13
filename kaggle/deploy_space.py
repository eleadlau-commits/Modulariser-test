"""Upload the GGUF to a private model repo and (re)deploy the free CPU Space."""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("gguf")
ap.add_argument("--run-id", required=True)
a = ap.parse_args()
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
user = api.whoami()["name"]
model_repo = f"{user}/modulariser-test-gguf"
space = f"{user}/modulariser-test"
api.create_repo(model_repo, private=True, exist_ok=True)
api.upload_file(path_or_fileobj=a.gguf, path_in_repo="model-q4_k_m.gguf", repo_id=model_repo,
                commit_message=f"GGUF from run {a.run_id}")
api.create_repo(space, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
api.add_space_secret(space, "HF_TOKEN", token)
api.add_space_variable(space, "MODEL_REPO", model_repo)
api.upload_folder(folder_path=str(Path(__file__).parent.parent / "space"), repo_id=space, repo_type="space",
                  commit_message=f"deploy run {a.run_id}")
api.restart_space(space)
print("space:", f"https://huggingface.co/spaces/{space}")
