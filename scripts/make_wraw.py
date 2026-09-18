"""Create data/W.raw (real BF16 weights, raw, back to back) from a safetensors model directory,
so that kernbench.py and the tests run without the 903 MB file from the development machine.
usage: python scripts/make_wraw.py [--source models/Qwen3-4B] [--mb 903]"""
import os, glob, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--source", default=f"{ROOT}/models/Qwen3-4B")
ap.add_argument("--mb", type=int, default=903)
a = ap.parse_args()
target = os.path.join(ROOT, "data", "W.raw"); os.makedirs(os.path.dirname(target), exist_ok=True)
remaining = a.mb * 1000 * 1000
with open(target, "wb") as out:
    for f in sorted(glob.glob(os.path.join(a.source, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if remaining <= 0: break
                if "proj.weight" not in k: continue
                w = sf.get_tensor(k)
                if w.dtype != torch.bfloat16 or w.dim() != 2: continue
                b = w.contiguous().view(torch.int16).numpy().tobytes()[:remaining]
                out.write(b); remaining -= len(b)
        if remaining <= 0: break
print(f"{target}: {os.path.getsize(target)/1e6:.0f} MB")
