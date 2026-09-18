"""Download the models used by the scripts into models/: a BF16 base model and, optionally, its DFloat11 checkpoint.
usage: python scripts/download_models.py [Qwen/Qwen3-4B] [--df11 DFloat11/Qwen3-4B-DF11]"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="Qwen/Qwen3-4B")
ap.add_argument("--df11", default=None, help="DFloat11 checkpoint, e.g. DFloat11/Qwen3-4B-DF11")
a = ap.parse_args()
PATTERNS = ["*.safetensors", "*.json", "*.txt", "merges.txt", "vocab.json"]

p = snapshot_download(a.model, allow_patterns=PATTERNS, local_dir=f"{ROOT}/models/{a.model.split('/')[-1]}")
print("model:", p)
if a.df11:
    q = snapshot_download(a.df11, local_dir=f"{ROOT}/models/{a.df11.split('/')[-1]}")
    cfg = json.load(open(os.path.join(q, "config.json")))
    base = cfg.get("_name_or_path") or (cfg.get("dfloat11_config") or {}).get("original_model")
    print("DFloat11:", q, "(base model according to its config:", base, ")")
