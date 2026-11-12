"""Upload a trained adapter to a Hugging Face model repo (one revision per run)."""
import argparse
import os

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("adapter")
ap.add_argument("--run-id", required=True)
ap.add_argument("--models", required=True)
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repo = f"{user}/modulariser-test-adapter"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=a.adapter, repo_id=repo,
                  commit_message=f"adapter from run {a.run_id} on {a.models}")
print("uploaded to", repo)
